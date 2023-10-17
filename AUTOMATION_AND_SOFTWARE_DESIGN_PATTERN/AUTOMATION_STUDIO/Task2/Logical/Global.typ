
TYPE
	Sys_Handler : 	STRUCT
		Initialize:BOOL:=FALSE;
		Run:BOOL:=FALSE;
		On_Phase_Stop:BOOL:=FALSE;
	END_STRUCT;

	Sub_Sys_Handler : 	STRUCT
		Initialize:BOOL:=FALSE;
		Enable:BOOL:=FALSE;
		Disable:BOOL:=FALSE;
		On_Phase_Stop:BOOL:=FALSE;
	END_STRUCT;

END_TYPE
