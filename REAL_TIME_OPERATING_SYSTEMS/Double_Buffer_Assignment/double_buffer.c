/*
 ============================================================================
 Name        : double_buffer.c
 Author      : Daniele Simonazzi - Federico Fabbri - Jacopo Merlo Pich
 Version     :
 Description : Double buffer assignment
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

// CONSTANTS AND MACROS
// for readability
#define N_THREADS 3
#define FOREVER for(;;)

// DEFINITIONS OF NEW DATA TYPES
// for readability
typedef char thread_name_t[10];
typedef enum {EMPTY, HALF, FULL} state_t;

typedef struct {
	//mutex lock
    pthread_mutex_t m;
    //numbers waiting
    int nw1,nw2;
    //counters to keep track of items produced/consumed by p1, p2, c
    int n_p1, n_p2, n_c;

    //condition variables
    pthread_cond_t one,two;
    // state variables
    state_t statevar;

} monitor_t;

// GLOBAL VARIABLES
// the monitor should be defined as a global variable
monitor_t mon;
int buffer[2] = {0,0};

//  MONITOR API
void push_int(monitor_t *mon, int value);
void push_pair(monitor_t *mon, int value1, int value2);
int fetch(monitor_t *mon);
int produce_value();
void monitor_init(monitor_t *mon);
void monitor_destroy(monitor_t *mon);
void *p_int(void *arg);
void *p_pair(void *arg);
void *p_consume(void *arg);
// IMPLEMENTATION OF MONITOR API
void push_int(monitor_t *mon, int value) {

	pthread_mutex_lock(&mon->m);

		mon->nw1++;

		while(mon->statevar==FULL || mon->nw2 !=0) {
			printf("\nP1 --- trying to produce\n\n");
			pthread_cond_wait(&mon->one,&mon->m);
		}

		mon->nw1--;
		mon->n_p1++;
		if(buffer[0]==0) buffer[0]=value;
		else buffer[1]=value;

		if(mon->statevar==EMPTY) mon->statevar=HALF;
		else mon->statevar=FULL;

		printf("\nP1 --- produced the number: [%d]\n",value);
		printf("\nP1 --- Final buffer: [%d , %d]\n\n", buffer[0], buffer[1]);

	pthread_mutex_unlock(&mon->m);
}

void push_pair(monitor_t *mon, int value1, int value2) {

	pthread_mutex_lock(&mon->m);

		mon->nw2++;

		while(mon->statevar!=EMPTY){
			printf("\nP2 --- trying to produce\n\n");
			pthread_cond_wait(&mon->two,&mon->m);
		}

		mon->nw2--;
		mon->n_p2+=2;

		buffer[0]=value1;
		buffer[1]=value2;

		mon->statevar=FULL;

		printf("\nP2 --- produced the pair of numbers: [%d , %d]\n",value1, value2);
		printf("\nP2 --- Final buffer: [%d , %d]\n\n", buffer[0], buffer[1]);

	pthread_mutex_unlock(&mon->m);
}

int fetch(monitor_t *mon) {

	pthread_mutex_lock(&mon->m);

		int val = 0;
		if(mon->statevar == EMPTY) {
			if(mon->nw2 != 0) pthread_cond_signal(&mon->two);
			else if(mon->nw1 != 0) pthread_cond_signal(&mon->one);
		}
		else if(mon->statevar == FULL){
			val = buffer[0];
			buffer[0]=0;
			mon->n_c++;
			mon->statevar=HALF;
			if(mon->nw1 != 0 && mon->nw2 == 0) pthread_cond_signal(&mon->one);
		}
		else if(mon->statevar==HALF){
			if (buffer[0]==0) {
				val = buffer[1];
				buffer[1]=0;
				mon->n_c++;
				mon->statevar=EMPTY;
				}

			else {
				val = buffer[0];
				buffer[0]=0;
				mon->n_c++;
				mon->statevar=EMPTY;
			}
			if(mon->nw2 != 0) pthread_cond_signal(&mon->two);
			else if(mon->nw1 != 0) pthread_cond_signal(&mon->one);
		}

		printf("\nC --- Final buffer: [%d , %d]\n\n", buffer[0], buffer[1]);
		printf("\n --- Occurencies (P1 P2 C): (%d %d %d)\n", mon->n_p1, mon->n_p2, mon->n_c);

	pthread_mutex_unlock(&mon->m);
	return val;
}

void monitor_init(monitor_t *mon) {
	// set initial value of monitor data structures, state variables, mutexes, counters, etc.
    // typically can use default attributes for monitor mutex and condvars
    pthread_mutex_init(&mon->m,NULL);
    pthread_cond_init(&mon->one,NULL);
    pthread_cond_init(&mon->two,NULL);
    // set all condvar counters to 0
    mon->nw1=0;
    mon->nw2=0;
    mon->n_p1=0;
    mon->n_p2=0;
    mon->n_c=0;

    // initialize whatever other structures
    mon->statevar = EMPTY;

}

void monitor_destroy(monitor_t *mon) {
    // set initial value of monitor data structures, state variables, mutexes, counters, etc.
    pthread_cond_destroy(&mon->one);
    pthread_cond_destroy(&mon->two);
    pthread_mutex_destroy(&mon->m);

}

// MAIN FUNCTION
int main(void) {
    // thread management data structures
    pthread_t P1,P2,C;

    // initialize monitor data strcture before creating the threads
	monitor_init(&mon);


    pthread_create(&P1, NULL, p_int, "P1");
    pthread_create(&P2, NULL, p_pair, "P2");
    pthread_create(&C, NULL, p_consume, "C");

    pthread_join(P1, NULL);
    pthread_join(P2, NULL);
    pthread_join(C, NULL);

    // free OS resources occupied by the monitor after creating the threads
    monitor_destroy(&mon);

    return EXIT_SUCCESS;

}

void *p_int(void *arg) {

	FOREVER {

		int value = produce_value();
		push_int(&mon, value);

		sleep(1);

	}

	pthread_exit(NULL);

}

void *p_pair(void *arg) {

	FOREVER {

		int value1 = produce_value();
		int value2 = produce_value();
		push_pair(&mon, value1, value2);

		sleep(3);

	}

	pthread_exit(NULL);

}

void *p_consume(void *arg) {

	FOREVER {

		int item_c = fetch(&mon);
		printf("\nC --- consumed number: [%d]\n\n",item_c);

		sleep(1);

	}

	pthread_exit(NULL);

}

int produce_value(){

	int value=rand()%100+1;
	return value;

}
