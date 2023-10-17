
FUNCTION_BLOCK gAxis (*TODO: Add your comment here*) (*$GROUP=User,$CAT=User,$GROUPICON=User.png,$CATICON=User.png*)
	VAR_INPUT
		Master : UDINT;
		Slave : UDINT;
		AxisOperation : INT;
		AxisCommand : WORD := 777;
		Distance : REAL;
		Position : REAL;
		Velocity : REAL;
		Acceleration : REAL;
		Deceleration : REAL;
		Jerk : REAL;
		Direction : USINT;
		ArraySize : INT;
		Scale : REAL;
		Offset : REAL;
		RatioNumerator : INT;
		RatioDenominator : INT;
		CamTableID : USINT;
		MasterScaling : REAL := 360;
		SlaveScaling : REAL := 360;
		MasterOffset : REAL := 10;
		SlaveOffset : REAL := 0;
		StartMode : USINT := 1; (*mcRELATIVE*)
		AxisEnable_in : BOOL;
		Reset : BOOL;
	END_VAR
	VAR_OUTPUT
		ActualPosition : REAL;
		ActualVelocity : REAL;
		AxisEnable_out : BOOL;
		CommandExecuted : BOOL := FALSE;
		CommandAborted : BOOL;
		AxisStatus : WORD;
		AxisError : BOOL;
		AxisErrorID : WORD;
		AxisNotInizialized : BOOL;
	END_VAR
	VAR
		INIT : INT;
		RUN : INT;
		OldDistance : REAL;
		OldVelocity : REAL;
		OldCamTableID : USINT;
		AxisState : gAxis_StateType := NotPowered;
		gA_Power : MC_Power;
		gA_Home : MC_Home;
		gA_MoveVelocity : MC_MoveVelocity;
		gA_MoveAdditive : MC_MoveAdditive;
		gA_CamIn : MC_CamIn;
		gA_CamOut : MC_CamOut;
		gA_ReadActualPosition : MC_ReadActualPosition;
		gA_ReadActualVelocity : MC_ReadActualVelocity;
		gA_ReadAxisError : MC_ReadAxisError;
		gA_ReadStatus : MC_ReadStatus;
		gA_Stop : MC_Stop;
	END_VAR
END_FUNCTION_BLOCK
