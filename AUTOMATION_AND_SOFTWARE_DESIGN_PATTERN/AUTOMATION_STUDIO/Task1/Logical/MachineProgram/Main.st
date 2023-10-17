
PROGRAM _INIT
	(* Insert code here *)
	 
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)
	CASE state OF
		
		Machine_ready_to_be_initialized:
			IF Machine_Handler.Initialize THEN
				Conveyor_Handler.Initialize := TRUE;
				Cutter_Handler.Initialize := TRUE;
				
				state := Machine_initializing;
			END_IF;
			
		Machine_initializing:
			IF NOT Conveyor_Handler.Initialize AND NOT Cutter_Handler.Initialize THEN
				Machine_Handler.Initialize := FALSE;
				
				state := Machine_ready_to_run;
			END_IF;
			
		Machine_ready_to_run:
			IF Machine_Handler.Run THEN
				Conveyor_Handler.Enable := TRUE;
				
				state := Conveyor_Enabling;
			END_IF;
			
		Conveyor_Enabling:
			IF Machine_Handler.On_Phase_Stop THEN
				Conveyor_Handler.On_Phase_Stop := TRUE;
				Cutter_Handler.On_Phase_Stop := TRUE;
				
				state := Machine_on_phase_stopping;
				
			ELSIF NOT Conveyor_Handler.Enable AND Trigger_logical THEN
				Conveyor_Handler.Disable := TRUE;
				
				state := Conveyor_Disabling;
			END_IF;
			
		Conveyor_Disabling:
			IF NOT Conveyor_Handler.Disable THEN
				Cutter_Handler.Enable := TRUE;
				
				state := Cutter_Enabling;
			END_IF;
			
		Cutter_Enabling:
			IF NOT Cutter_Handler.Enable THEN
				Cutter_Handler.Disable := TRUE;
				
				state := Cutter_Disabling;
			END_IF;
			
		Cutter_Disabling:
			IF Machine_Handler.On_Phase_Stop THEN
				Conveyor_Handler.On_Phase_Stop := TRUE;
				Cutter_Handler.On_Phase_Stop := TRUE;
				
				state := Machine_on_phase_stopping;
				
			ELSIF NOT Cutter_Handler.Disable THEN
				Conveyor_Handler.Enable :=  TRUE;
				
				state := Conveyor_Enabling;
			END_IF;
			
		Machine_on_phase_stopping:
			IF NOT Machine_Handler.On_Phase_Stop THEN
				Conveyor_Handler.On_Phase_Stop := FALSE;
				Cutter_Handler.On_Phase_Stop := FALSE;
				
				state := Machine_ready_to_run;
			END_IF;
	END_CASE;
	 
END_PROGRAM

PROGRAM _EXIT
	(* Insert code here *)

	 
END_PROGRAM

