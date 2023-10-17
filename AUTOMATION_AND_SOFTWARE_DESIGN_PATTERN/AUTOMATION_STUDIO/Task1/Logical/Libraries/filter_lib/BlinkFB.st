
(* TODO: Add your comment here *)
FUNCTION_BLOCK BlinkFB
	IF Clock = 0 THEN
		LED := NOT LED;
		Clock := Delay;
			
	ELSE
		Clock := Clock -1;
	END_IF
END_FUNCTION_BLOCK
