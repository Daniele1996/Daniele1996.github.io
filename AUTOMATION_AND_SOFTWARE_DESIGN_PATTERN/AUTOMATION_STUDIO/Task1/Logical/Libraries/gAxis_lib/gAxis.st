
(* TODO: Add your comment here *)
FUNCTION_BLOCK gAxis
	
	CASE AxisState OF
		
		NotPowered:
			IF AxisEnable_in THEN
				gA_Power.Enable := TRUE;
				AxisState := WaitPower;
			END_IF;
			
		WaitPower:
			IF gA_Power.Status THEN
				gA_Home.Execute := TRUE;
				AxisState := Powered;
			END_IF
			
		Powered:
			IF gA_Home.Done THEN
				gA_Home.Execute := FALSE;
				AxisEnable_out := FALSE;
				AxisState := Ready;
			END_IF;
			
		Ready:
			IF AxisCommand = AXIS_MOVE_VELOCITY THEN
				CommandExecuted := FALSE;
				gA_MoveVelocity.Velocity := Velocity;
				gA_MoveVelocity.Execute := TRUE;
				AxisState := MoveVelocity;				
			END_IF;
			
			IF AxisCommand = AXIS_MOVE_ADDITIVE THEN
				CommandExecuted := FALSE;
				gA_MoveAdditive.Distance := Distance;
				gA_MoveAdditive.Execute := TRUE;
				AxisState := MoveAdditive;	
			END_IF;
			
			IF AxisCommand = AXIS_STOP THEN
				CommandExecuted := FALSE;
				gA_Stop.Execute := TRUE;
				AxisState := Stop;
			END_IF;
	
		MoveVelocity:
			IF gA_MoveVelocity.InVelocity THEN
				OldVelocity := Velocity;
				gA_MoveVelocity.Execute := FALSE;
				CommandExecuted := TRUE;
				AxisState := CheckVelocity;				
			END_IF;

			IF AxisCommand <> AXIS_MOVE_VELOCITY THEN
				gA_MoveVelocity.Execute := FALSE;
				CommandExecuted := FALSE;
				AxisState := Ready;		
			END_IF;
			
		CheckVelocity:
			IF Velocity <> OldVelocity THEN
				OldVelocity := Velocity;
				gA_MoveVelocity.Velocity := Velocity;
				gA_MoveVelocity.Execute := TRUE;
				CommandExecuted := FALSE;
				AxisState := MoveVelocity;			
			END_IF;
		
			IF AxisCommand <> AXIS_MOVE_VELOCITY THEN
				gA_MoveVelocity.Execute := FALSE;
				CommandExecuted := FALSE;
				AxisState := Ready;		
			END_IF;
		
		MoveAdditive:
			IF gA_MoveAdditive.Done THEN
				OldDistance := Distance;
				gA_MoveAdditive.Execute := FALSE;
				CommandExecuted := TRUE;
				AxisState := CheckPosition;
			END_IF;

			IF AxisCommand <> AXIS_MOVE_ADDITIVE THEN
				gA_MoveAdditive.Execute := FALSE;
				CommandExecuted := FALSE;
				AxisState := Ready;		
			END_IF;
			
		CheckPosition:
			IF Distance <> OldDistance THEN
				OldDistance := Distance;
				gA_MoveAdditive.Distance := Distance;
				gA_MoveAdditive.Execute := TRUE;
				CommandExecuted := FALSE;
				AxisState := MoveAdditive;			
			END_IF;
		
			IF AxisCommand <> AXIS_MOVE_ADDITIVE THEN
				gA_MoveAdditive.Execute := FALSE;
				CommandExecuted := FALSE;
				AxisState := Ready;		
			END_IF;
		
		Stop:
			IF (gA_Stop.Done) THEN
				CommandExecuted := TRUE;
			END_IF
           
			IF (AxisCommand <> AXIS_STOP ) THEN
				gA_Stop.Execute := FALSE;
				CommandExecuted := FALSE;
				
				AxisState := Ready;
			END_IF
		
	END_CASE;
	
	AxisEnable_in := AxisEnable_out;
	
	gA_Power.Axis := Slave;
	gA_Power();
	
	gA_Home.Axis := Slave;
	gA_Home.Position := Position;
	gA_Home();
	
	gA_MoveVelocity.Axis := Slave;
	gA_MoveVelocity.Velocity := Velocity;
	gA_MoveVelocity.Acceleration := Acceleration;
	gA_MoveVelocity.Deceleration := Deceleration;
	gA_MoveVelocity();
	
	IF OldDistance = 0 THEN
		OldDistance := Distance;
	END_IF;
	
	gA_MoveAdditive.Axis := Slave;
	gA_MoveAdditive.Distance := Distance;
	gA_MoveAdditive.Velocity := Velocity;
	gA_MoveAdditive.Acceleration := Acceleration;
	gA_MoveAdditive.Deceleration := Deceleration;
	gA_MoveAdditive();	
	
	gA_ReadActualPosition.Axis := Slave;
	gA_ReadActualPosition.Enable := TRUE;
	gA_ReadActualPosition();
	
	ActualPosition := gA_ReadActualPosition.Position;
	
	gA_ReadActualVelocity.Axis := Slave;
	gA_ReadActualVelocity.Enable := TRUE;
	gA_ReadActualVelocity();
	
	ActualVelocity := gA_ReadActualVelocity.Velocity;
	
	gA_Stop.Axis := Slave;
	gA_Stop.Deceleration := Deceleration;
	gA_Stop();
	
	gA_ReadStatus.Enable := TRUE;
	gA_ReadStatus.Axis := Slave;
	gA_ReadStatus();
	
	IF gA_ReadStatus.StandStill THEN
		AxisStatus := STANDSTILL;
	END_IF;
	
	IF gA_ReadStatus.DiscreteMotion THEN
		AxisStatus := DISCRETE_MOTION;
	END_IF;
	
	IF gA_ReadStatus.ContinuousMotion THEN
		AxisStatus := CONTINOUS_MOTION;
	END_IF;
	
END_FUNCTION_BLOCK
