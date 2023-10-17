/*
 ============================================================================
 Name        : penguins.c
 Author      : Daniele Simonazzi - Federico Fabbri - Jacopo Merlo Pich
 Version     :
 Description : Penguins assignment
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <ctype.h>

#define FOREVER while(!time_ended)
#define N_PENGUINS 6

/* the following values are just examples of the possible duration
 * of each action and of the simulation: feel free to change them */
#define BIG_SLEEP_TIME 4
#define SMALL_SLEEP_TIME 10
#define WALK_TIME 2
#define JUMP_TIME 1
#define SWIM_TIME 6
#define EXIT_TIME 1
#define END_OF_TIME 120

typedef char name_t[20];
typedef enum {FALSE, TRUE} boolean;

int waiting=0, swimming=0, n_pen=0;
int n_p_trips[N_PENGUINS];
time_t big_bang;
boolean time_ended=FALSE;

sem_t semp1, semp2, semP;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

void do_action(char *thread_name, char *action_name, int max_delay) {
	int delay=rand()%max_delay+1;
	printf("[%4.0f]\t%s: %s (%d) started\n", difftime(time(NULL),big_bang), thread_name, action_name, delay);
	sleep(delay);
	printf("[%4.0f]\t%s: %s (%d) ended\n", difftime(time(NULL),big_bang), thread_name, action_name, delay);
}
void *big_penguin(void *thread_name) {

    FOREVER {

		do_action(thread_name, "rest on the ice", BIG_SLEEP_TIME);

		pthread_mutex_lock(&m);
		n_pen = waiting;
		waiting = 0;
		pthread_mutex_unlock(&m);

		printf("\n\t---- %d READY LITTLE PENGUINS ----\n\n", n_pen);

		do_action(thread_name, "walk to the water", WALK_TIME);

        //Scenario 1: # of ready little penguins != 0
		if(n_pen != 0){
			for(int i=0; i<n_pen; i++)
				sem_post(&semp1);
			sem_wait(&semP);

			do_action(thread_name, "jump in the water", JUMP_TIME);

			for(int i=0; i<n_pen; i++)
				sem_post(&semp2);

			do_action(thread_name, "swim", SWIM_TIME);

			sem_wait(&semP);

			do_action(thread_name, "go out the water", EXIT_TIME);

		}

        //Scenario 2: # of ready little penguins == 0
		else  {
			do_action(thread_name, "jumping in the water", JUMP_TIME);
			do_action(thread_name, "swim", SWIM_TIME);
			do_action(thread_name, "go out the water", EXIT_TIME);
		}

	    n_p_trips[0]++;
        printf("\n\t---- BIG PENGUIN'S CYCLE RESTART ----\n\n");

	}

	printf("Terminating big penguin.\n");

    //Printing final statistics
	printf("\n\tBig penguin has swum %d times\n\n",  n_p_trips[0]);
	for (int i = 1; i<N_PENGUINS;i++)
			printf("\n\tLittle penguin %d have swum %d times\n\n", i, n_p_trips[i]);
	pthread_exit(NULL);
}

void *little_penguin(void *thread_name) {

    FOREVER {

        /*
        We use the following snippet of code in order to identify the index of the current thread,
        for statistics purposes
        */
		long index_p;
		char *p = thread_name;
		while (*p) {
		    if (isdigit(*p)) {
		        index_p = strtol(p, &p, 10);
		    }
		    else
		    	p++;
		}

		do_action(thread_name, "rest on the ice", SMALL_SLEEP_TIME);

		pthread_mutex_lock(&m);
		waiting++;
		pthread_mutex_unlock(&m);

		sem_wait(&semp1); //Wait for big penguin signal

		do_action(thread_name, "walk to the water", WALK_TIME);
        do_action(thread_name, "jump in the water", JUMP_TIME);


		pthread_mutex_lock(&m);
		n_p_trips[index_p]++;
		swimming++;
        //if it's the last one, signal sent to the big penguin
		if (swimming == n_pen)
            sem_post(&semP);
		pthread_mutex_unlock(&m);

		sem_wait(&semp2); //Wait for the big one to start swimming

		do_action(thread_name, "swim", SWIM_TIME);

		do_action(thread_name, "go out the water", EXIT_TIME);

		pthread_mutex_lock(&m);
		swimming--;
        //if it's the last one, signal sent to the big penguin
		if(swimming==0)
			sem_post(&semP);
		pthread_mutex_unlock(&m);

	}
	printf("Terminating %s.\n", (char*) thread_name);
	pthread_exit(NULL);
}

int main(void) {
	pthread_t penguin_id[N_PENGUINS];
	name_t penguin_name[N_PENGUINS];
	int i;


	sem_init(&semP,0,0);
	sem_init(&semp1,0,0);
	sem_init(&semp2,0,0);

	time(&big_bang);

	sprintf(penguin_name[0],"big penguin");
	pthread_create(penguin_id,NULL,big_penguin,penguin_name);
	for(i=1;i<N_PENGUINS;i++) {
		n_p_trips[i]=0;
		sprintf(penguin_name[i],"little penguin #%d",i);
		pthread_create(penguin_id+i,NULL,little_penguin,penguin_name+i);
	}
	sleep(END_OF_TIME);
	time_ended=TRUE;

	for(i=0;i<N_PENGUINS;i++) {
		pthread_join(penguin_id[i],NULL);
	}
	pthread_mutex_destroy(&m);
	sem_destroy(&semp1);
	sem_destroy(&semp2);
	sem_destroy(&semP);


	return EXIT_SUCCESS;
}
