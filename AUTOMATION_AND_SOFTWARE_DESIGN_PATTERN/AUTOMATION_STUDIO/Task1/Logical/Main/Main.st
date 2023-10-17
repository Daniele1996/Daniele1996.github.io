
PROGRAM _INIT
	(* Insert code here *)
	 
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)
	
	CASE state OF
		Main_ready_to_be_initialized:
			IF Init_logical THEN
				Machine_Handler.Initialize := TRUE;
				
				state := Main_initializing;
			END_IF;
			
		Main_initializing:
			IF NOT Machine_Handler.Initialize THEN

				state := Main_ready_to_run;
			END_IF;
			
		Main_ready_to_run:
			IF Start_logical THEN
				Machine_Handler.Run := TRUE;
				
				state := Main_running;
			END_IF;
			
		Main_running:
			IF On_Phase_Stop_logical THEN
				Machine_Handler.Run := FALSE;
				Machine_Handler.On_Phase_Stop := TRUE;
				
				state := Main_on_phase_stopping;
			END_IF;
			
		Main_on_phase_stopping:
			IF NOT On_Phase_Stop_logical THEN
				Machine_Handler.On_Phase_Stop := FALSE;
				
				state := Main_ready_to_run;
			END_IF;
	END_CASE;

	 
END_PROGRAM

PROGRAM _EXIT
	(* Insert code here *)
	 
END_PROGRAM

