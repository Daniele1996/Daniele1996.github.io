
PROGRAM _INIT
	(* Insert code here *)
	
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)
	CASE state OF
		Conveyor_ready_to_be_initialized:
			IF Conveyor_Handler.Initialize THEN
				Axis_Enable := TRUE;
				
				state := Conveyor_initializing;
			END_IF;
			
		Conveyor_initializing:
			IF NOT Axis_Enable THEN
				Conveyor_Handler.Initialize := FALSE;
				
				state := Conveyor_ready_to_be_enabled;
			END_IF;
			
		Conveyor_ready_to_be_enabled:
			IF Conveyor_Handler.Enable THEN
				Axis_Command := AXIS_MOVE_VELOCITY;
				
				state := Conveyor_enabling;
			END_IF;
			
		Conveyor_enabling:
			IF Axis_Command_Executed THEN
				Conveyor_Handler.Enable := FALSE;
				
				state := Conveyor_ready_to_be_disabled;
			END_IF;
			
		Conveyor_ready_to_be_disabled:
			IF Conveyor_Handler.On_Phase_Stop THEN
				Axis_Command := AXIS_STOP;
				
				state := Conveyor_on_phase_stopping;
				
			ELSIF Conveyor_Handler.Disable THEN
				Axis_Command := AXIS_MOVE_ADDITIVE;
				
				state := Conveyor_disabling;
			END_IF;
			
		Conveyor_disabling:
			IF Axis_Command_Executed THEN
				Conveyor_Handler.Disable := FALSE;
				
				state := Conveyor_ready_to_be_enabled;
			END_IF;
		
		Conveyor_on_phase_stopping:
			IF NOT Conveyor_Handler.On_Phase_Stop THEN
				
				state := Conveyor_ready_to_be_enabled;
			END_IF;
	END_CASE;
	
	Conveyor.Slave	:= ADR(gConveyor);
 
	Conveyor.AxisEnable_in := Axis_Enable;
	Conveyor.AxisCommand := Axis_Command;
	
	Conveyor.Distance	:= CONVEYOR_DISTANCE;
	Conveyor.Velocity	:= CONVEYOR_VELOCITY;
	Conveyor.Acceleration	:= CONVEYOR_ACCELLERATION;
	Conveyor.Deceleration	:= CONVEYOR_DEACCELLERATION;
	
	Conveyor();
	
	Axis_Enable := Conveyor.AxisEnable_out;
	Axis_Command_Executed := Conveyor.CommandExecuted;
	CONVEYOR_POSITION := Conveyor.ActualPosition;
	CONVEYOR_ACTUAL_VELOCITY := REAL_TO_INT(Conveyor.ActualVelocity);
	
	ComTCP_SimOutputs.Shaft_0_ACT_POS := CONVEYOR_POSITION;
END_PROGRAM

PROGRAM _EXIT
	(* Insert code here *)
	 
END_PROGRAM

