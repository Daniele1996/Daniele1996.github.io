
PROGRAM _INIT
	(* Insert code here *)
	 
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)
	
	CASE state OF
		
		Cutter_ready_to_be_initialized:
			IF Cutter_Handler.Initialize THEN
				Axis_Enable := TRUE;
				CamTableSelect.Execute := TRUE;
				
				state := Cutter_initializing;
			END_IF;
			
		Cutter_initializing:
			IF NOT Axis_Enable AND CamTableSelect.Done THEN
				Cutter_Handler.Initialize := FALSE;
				CamTableID := CamTableSelect.CamTableID;
				CamTableSelect.Execute := FALSE;
				
				state := Cutter_ready_to_be_enabled;
			END_IF;
			
		Cutter_ready_to_be_enabled:
			IF Cutter_Handler.Enable THEN
				Axis_Command := AXIS_CAM_IN;
				
				state := Cutter_enabling;
			END_IF;
			
		Cutter_enabling:
			IF Axis_Command_Executed THEN
				Axis_Command := AXIS_NULL_COMMAND;
				Cutter_Handler.Enable := FALSE;
				
				state := Cutter_ready_to_be_enabled;
			END_IF;

	END_CASE;
	 
	CamTableSelect.Master := ADR(gConveyor);
	CamTableSelect.Slave := ADR(gKnife);
	CamTableSelect.CamTable := 'Cam';
	CamTableSelect.Periodic := mcNON_PERIODIC;
	CamTableSelect();

	
	
	Knife.Master	:= ADR(gConveyor);
	Knife.Slave	:= ADR(gKnife);
	
	Knife.AxisEnable_in := Axis_Enable;
	Knife.AxisCommand := Axis_Command;
	
	Knife.CamTableID := CamTableID;
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

