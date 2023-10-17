
PROGRAM _INIT
	(* Insert code here *)
	 
END_PROGRAM

PROGRAM _CYCLIC
	(* Insert code here *)

	DI1_filter.Signal := DI1;
	DI1_filter.ActivationDelay := DI1_delay;
	DI1_filter.DeactivationDelay := DI1_delay;
	DI1_filter();
	
	DI3_filter.Signal := DI3;
	DI3_filter.ActivationDelay := DI3_delay;
	DI3_filter.DeactivationDelay := DI3_delay;
	DI3_filter();
	 
	DI4_filter.Signal := DI4;
	DI4_filter.ActivationDelay := DI4_delay;
	DI4_filter.DeactivationDelay := DI4_delay;
	DI4_filter();
	
	DI5_filter.Signal := DI5;
	DI5_filter.ActivationDelay := DI5_delay;
	DI5_filter.DeactivationDelay := DI5_delay;
	DI5_filter(); 
	
	DI6_filter.Signal := DI6;
	DI6_filter.ActivationDelay := DI6_delay;
	DI6_filter.DeactivationDelay := DI6_delay;
	DI6_filter();

	IF SIM_ON_HMI THEN
		
		Init_logical := DI1_filter.DelayedSignal OR INIT_HMI;
		Start_logical := DI3_filter.DelayedSignal OR START_HMI;
		On_Phase_Stop_logical := DI4_filter.DelayedSignal OR STOP_HMI;
		Trigger_logical := DI5_filter.DelayedSignal OR TRIGGER_HMI;
		Reset_logical := DI6_filter.DelayedSignal OR RESET_HMI;
		
		CONVEYOR_VELOCITY := UINT_TO_REAL(CONVEYOR_SPEED_HMI)*CONVEYOR_MAX_VELOCITY/(65535);
		CUTTER_VELOCITY := UINT_TO_REAL(CUTTER_SPEED_HMI)*CUTTER_MAX_VELOCITY/(65535);
	ELSE
		Init_logical := DI1_filter.DelayedSignal;
		Start_logical := DI3_filter.DelayedSignal;
		On_Phase_Stop_logical := DI4_filter.DelayedSignal;
		Trigger_logical := DI5_filter.DelayedSignal;
		Reset_logical := DI6_filter.DelayedSignal;
		
		CONVEYOR_VELOCITY := CONVEYOR_MAX_VELOCITY/(32767)*P1;
		CUTTER_VELOCITY := CUTTER_MAX_VELOCITY/(32767)*P2;
	END_IF
	
END_PROGRAM

PROGRAM _EXIT
	(* Insert code here *)
	 
END_PROGRAM

