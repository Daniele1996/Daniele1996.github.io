
{REDUND_ERROR} {REDUND_UNREPLICABLE} FUNCTION_BLOCK filter (*TODO: Add your comment here*) (*$GROUP=User,$CAT=User,$GROUPICON=User.png,$CATICON=User.png*)
	VAR_INPUT
		Signal : {REDUND_UNREPLICABLE} BOOL;
		ActivationDelay : {REDUND_UNREPLICABLE} UDINT;
		DeactivationDelay : {REDUND_UNREPLICABLE} UDINT;
	END_VAR
	VAR_OUTPUT
		DelayedSignal : {REDUND_UNREPLICABLE} BOOL;
	END_VAR
	VAR
		Delay : {REDUND_UNREPLICABLE} UDINT;
	END_VAR
END_FUNCTION_BLOCK

{REDUND_ERROR} {REDUND_UNREPLICABLE} FUNCTION_BLOCK BlinkFB (*TODO: Add your comment here*) (*$GROUP=User,$CAT=User,$GROUPICON=User.png,$CATICON=User.png*)
	VAR_INPUT RETAIN
		Delay : {REDUND_UNREPLICABLE} INT := 5;
	END_VAR
	VAR_IN_OUT
		LED : BOOL;
	END_VAR
	VAR
		Clock : {REDUND_UNREPLICABLE} INT;
	END_VAR
END_FUNCTION_BLOCK
