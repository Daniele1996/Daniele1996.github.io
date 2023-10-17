
PROGRAM _CYCLIC
	(* Insert code here *)
	
	IF Machine_Handler.Run THEN
		Led_Start := TRUE;
		Led_Stop := TRUE;
		
	ELSIF NOT Machine_Handler.Run AND NOT Machine_Handler.Initialize AND Init_logical AND NOT Machine_Handler.On_Phase_Stop THEN
		Blink(LED := Led_Start);
		Led_Stop := FALSE;
		
	ELSE
		Led_Start := FALSE;
		Led_Stop := FALSE;
		
	END_IF;
	
END_PROGRAM
