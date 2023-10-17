
PROGRAM _INIT
	(* Insert code here *)
	 
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)
	
	CASE state OF
		
		Cutter_ready_to_be_initialized:
			IF Cutter_Handler.Initialize THEN
				Axis_Enable := TRUE;
				
				state := Cutter_initializing;
			END_IF;
			
		Cutter_initializing:
			IF NOT Axis_Enable THEN
				Cutter_Handler.Initialize := FALSE;
				
				state := Cutter_ready_to_be_enabled;
			END_IF;
			
		Cutter_ready_to_be_enabled:
			IF Cutter_Handler.Enable THEN
				CUTTER_DISTANCE := BELT_POSITION;
				Axis_Command := AXIS_MOVE_ADDITIVE;
				
				state := Cutter_enabling;
			END_IF;
			
		Cutter_enabling:
			IF Axis_Command_Executed THEN
				Cutter_Handler.Enable := FALSE;
				
				state := Cutter_ready_to_be_disabled;
			END_IF;
			
		Cutter_ready_to_be_disabled:
			IF Cutter_Handler.Disable THEN
				CUTTER_DISTANCE := - BELT_POSITION;
				Axis_Command := AXIS_MOVE_ADDITIVE;
				
				state := Cutter_disabling;
			END_IF;
			
		Cutter_disabling:
			IF Axis_Command_Executed THEN
				Cutter_Handler.Disable := FALSE;
				
				state := Cutter_ready_to_be_enabled;
			END_IF;
		
	END_CASE;
	 
	Knife.Slave	:= ADR(gKnife);
	 
	Knife.AxisEnable_in := Axis_Enable;
	Knife.AxisCommand := Axis_Command;
	
	Knife.Distance	:= CUTTER_DISTANCE;
	Knife.Velocity := CUTTER_VELOCITY;
	Knife.Acceleration	:= CUTTER_ACCELLERATION;
	Knife.Deceleration	:= CUTTER_DEACCELLERATION;

	Knife();

	Axis_Enable := Knife.AxisEnable_out;
	Axis_Command_Executed := Knife.CommandExecuted;
	CUTTER_POSITION	:= Knife.ActualPosition;
	CUTTER_ACTUAL_VELOCITY:= Knife.ActualVelocity;
	
		
	ComTCP_SimOutputs.Shaft_1_ACT_POS := CUTTER_POSITION;
	
END_PROGRAM

PROGRAM _EXIT
	(* Insert code here *)
	 
END_PROGRAM

