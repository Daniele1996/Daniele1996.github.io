CoDeSys+�  �               
   Tesi @       Studenti @   2.3.9.59�   Marco Pierantoni @   ConfigExtension�          CommConfigEx7             CommConfigExEnd   ME�                  IB                    % QB                    %   ME_End   CM�      CM_End   CT�   ��������   CT_End   ConfigExtensionEnd?    @                                     �Q�d +    @      ��������             ��.H        ��   @   D   C:\Program Files (x86)\3S Software\CODESYS V2.3\Library\STANDARD.LIB          CONCAT               STR1               ��              STR2               ��                 CONCAT                                         �)�S  �   ����           CTD           M             ��           Variable for CD Edge Detection      CD            ��           Count Down on rising edge    LOAD            ��	           Load Start Value    PV           ��
           Start Value       Q            ��           Counter reached 0    CV           ��           Current Counter Value             �)�S  �   ����           CTU           M             ��            Variable for CU Edge Detection       CU            ��       
    Count Up    RESET            ��	           Reset Counter to 0    PV           ��
           Counter Limit       Q            ��           Counter reached the Limit    CV           ��           Current Counter Value             �)�S  �   ����           CTUD           MU             ��            Variable for CU Edge Detection    MD             ��            Variable for CD Edge Detection       CU            ��
       
    Count Up    CD            ��           Count Down    RESET            ��           Reset Counter to Null    LOAD            ��           Load Start Value    PV           ��           Start Value / Counter Limit       QU            ��           Counter reached Limit    QD            ��           Counter reached Null    CV           ��           Current Counter Value             �)�S  �   ����           DELETE               STR               ��              LEN           ��	              POS           ��
                 DELETE                                         �)�S  �   ����           F_TRIG           M             ��                 CLK            ��           Signal to detect       Q            ��	           Edge detected             �)�S  �   ����           FIND               STR1               ��	              STR2               ��
                 FIND                                     �)�S  �   ����           INSERT               STR1               ��	              STR2               ��
              POS           ��                 INSERT                                         �)�S  �   ����           LEFT               STR               ��              SIZE           ��                 LEFT                                         �)�S  �   ����           LEN               STR               ��                 LEN                                     �)�S  �   ����           MID               STR               ��              LEN           ��	              POS           ��
                 MID                                         �)�S  �   ����           R_TRIG           M             ��                 CLK            ��           Signal to detect       Q            ��	           Edge detected             �)�S  �   ����           REPLACE               STR1               ��	              STR2               ��
              L           ��              P           ��                 REPLACE                                         �)�S  �   ����           RIGHT               STR               ��              SIZE           ��                 RIGHT                                         �)�S  �   ����           RS               SET            ��              RESET1            ��	                 Q1            ��                       �)�S  �   ����           RTC           M             ��              DiffTime            ��                 EN            ��              PDT           ��                 Q            ��              CDT           ��                       �)�S  �   ����           SEMA           X             ��                 CLAIM            ��
              RELEASE            ��                 BUSY            ��                       �)�S  �   ����           SR               SET1            ��              RESET            ��                 Q1            ��                       �)�S  �   ����           TOF           M             ��           internal variable 	   StartTime            ��           internal variable       IN            ��       ?    starts timer with falling edge, resets timer with rising edge    PT           ��           time to pass, before Q is set       Q            ��       2    is FALSE, PT seconds after IN had a falling edge    ET           ��           elapsed time             �)�S  �   ����           TON           M             ��           internal variable 	   StartTime            ��           internal variable       IN            ��       ?    starts timer with rising edge, resets timer with falling edge    PT           ��           time to pass, before Q is set       Q            ��       0    is TRUE, PT seconds after IN had a rising edge    ET           ��           elapsed time             �)�S  �   ����           TP        	   StartTime            ��           internal variable       IN            ��       !    Trigger for Start of the Signal    PT           ��       '    The length of the High-Signal in 10ms       Q            ��           The pulse    ET           ��       &    The current phase of the High-Signal             �)�S  �   ����    B   C:\Program Files (x86)\3S Software\CODESYS V2.3\Library\IECSFC.LIB          SFCACTIONCONTROL     
      S_FF                 RS    ��              L_TMR                    TON    ��              D_TMR                    TON    ��              P_TRIG                 R_TRIG    ��              SD_TMR                    TON    ��              SD_FF                 RS    ��              DS_FF                 RS    ��              DS_TMR                    TON    ��              SL_FF                 RS    ��              SL_TMR                    TON    ��           
      N            ��           Non stored action qualifier    R0            ��       #    Overriding reset action qualifier    S0            ��           Set (stored) action qualifier    L            ��	           Time limited action qualifier    D            ��
           Time delayed action qualifier    P            ��           Pulse action qualifier    SD            ��       *    Stored and time delayed action qualifier    DS            ��       %    Delayed and stored action qualifier    SL            ��       *    Stored and time limited action qualifier    T           ��           Current time       Q            ��       1    Associated action is executed, if Q equals TRUE             �)�S  �    ����    G   C:\Program Files (x86)\3S Software\CODESYS V2.3\Library\ANALYZATION.LIB          ANALYZEEXPRESSION               InputExp            ��           	   DoAnalyze            ��              	   ExpResult            ��           	   OutString               ��                       �)�S  �    ����           APPENDERRORSTRING               strOld               ��              strNew               ��                 AppendErrorString                                         �)�S  �    ����        '          ASSEMBLY_PRG           OperationType            u               state           Assembly_ready_to_initialize       Assembly_States    u               BlockingCylinder                Generic_Device    u               BlockingCylinder_enable_request             u                BlockingCylinder_disable_request             u                BlockingCylinder_not_initialized             u                              Assembly_Handler                   Subsystem_Handler   u 
              Assembly_Data                   Data_Handler   u               ItemsSupply_Handler                   Subsystem_Handler   u               ItemsSupply_SubData                   Data_Handler   u               ItemsAssembling_Handler                   Subsystem_Handler   u               enable_BlockingCylinder            u %              BlockingCylinder_ActuatorFault            u &           Pure sensors (   EmptyCoverHouseInAssemblyStation_Logical            u )               ��d  @   ����           BRIDGE_ASSEMBLY_GDS                             ��d  @    ����           BRIDGE_DISTRIBUTION_GDS                             ��d  @    ����           BRIDGE_DRILLING_GDS                             ��d  @    ����           BRIDGE_INSPECTION_GSS                             ��d  @    ����           BRIDGE_ITEMSASSEMBLING_GDS                             ��d  @    ����           BRIDGE_ITEMSSUPPLY_GDS                             ��d  @    ����           BRIDGE_PICKANDPLACE_GDS                             ��d  @    ����           BRIDGE_PROCESSING_GDS                             ��d  @    ����           BRIDGE_TESTING_GDS                             ��d  @    ����           BUTTONS           Start_Button             n               Reset_Button             n               OnPhaseStop_Button             n                ImmediateStop_Button             n !              Filter_Start                Signal_Filter    n 0              Filter_Reset                Signal_Filter    n 1              Filter_OnPhaseStop                Signal_Filter    n 2              Filter_ImmediateStop                Signal_Filter    n 3                             Start            n               Reset            n               Stop           n               FreezeStopPuls           n               Init_HMI            n            	   Start_HMI            n               Stop_HMI            n            	   Reset_HMI            n               OnPhaseStop_HMI            n               ImmediateStop_HMI            n               ResetSignalsEnable            n               OnPhaseStop_Signal            n               ImmediateStop_Signal            n               Init_Logical            n &              Start_Logical            n '              Reset_Logical            n (              Stop_Logical            n )              OnPhaseStop_Logical            n *              ImmediateStop_Logical            n +               ��d  @    ����           DISTRIBUTION_PRG           OperationType            9               state       !    Distribution_ready_to_initialize       Distribution_States    9               counter            9               Cylinder                Generic_Device    9               Cylinder_enable_request             9               Cylinder_disable_request             9               Cylinder_not_initialized             9               RotaryMaker                Generic_Device    9               RotaryMaker_enable_request             9               RotaryMaker_disable_request             9               RotaryMaker_not_initialized             9               VacuumGenerator                Generic_Device    9 !              VacuumGenerator_enable_request             9 "              VacuumGenerator_disable_request             9 #              VacuumGenerator_not_initialized             9 $              ExpulsionAirVacuum                Generic_Device    9 &           !   ExpulsionAirVacuum_enable_request             9 '           "   ExpulsionAirVacuum_disable_request             9 (           "   ExpulsionAirVacuum_not_initialized             9 )                             Distribution_Handler                   Subsystem_Handler   9               Distribution_Data                   Data_Handler   9               enable_Cylinder            9 /              Cylinder_enabled            9 0              Cylinder_disabled            9 1              Cylinder_EnabledSensorFault            9 2              Cylinder_DisabledSensorFault            9 3              Cylinder_fault            9 4              Cylinder_ActuatorFault            9 5           RotaryMaker - DA_DF    enable_RotaryMaker            9 8              disable_RotaryMaker            9 9              RotaryMaker_enabled            9 :              RotaryMaker_disabled            9 ;              RotaryMaker_EnabledSensorFault            9 <              RotaryMaker_DisabledSensorFault            9 =              RotaryMaker_fault            9 >              RotaryMaker_ActuatorFault            9 ?           VacuumGenerator - SA_SAF    enable_VacuumGenerator            9 B              VacuumGenerator_enabled            9 C           "   VacuumGenerator_EnabledSensorFault            9 D              VacuumGenerator_fault            9 E              VacuumGenerator_ActuatorFault            9 F       #    GDs - Sensors - Logical variables    enable_ExpulsionAirVacuum            9 I               ExpulsionAirVacuum_ActuatorFault            9 J           Pure sensors    EmptyWarehouse_Logical           9 M               ��d  @   ����           DISTTEST_PRG           state_DisTest           DisTest_ready_to_initialize       DisTest_States    C 4                             Memory_Data   	                        Data_Handler           C               Distribution_index         C 	              Testing_index         C 
              Rotary_index         C               Inspection_index         C               Drilling_index         C               Expelling_index         C               PickandPlace_index         C               Supply_index         C               DistTest_Handler                 System_Handler   C               Distribution_Handler                   Subsystem_Handler   C               Distribution_Data                   Data_Handler   C               Testing_Handler                   Subsystem_Handler   C "              Testing_Data                   Data_Handler   C #              Testing_ready_to_send            C (              Processing_ready_to_receive            C -               ��d  @    ����           DRILLINGUNIT_PRG           OperationType            N               state           Drilling_ready_to_initialize       DrillingUnit_States    N               Holding                Generic_Device    N               Holding_enable_request             N               Holding_disable_request             N               Holding_not_initialized             N               Drill_Machine                Generic_Device    N               Drill_Machine_enable_request             N               Drill_Machine_disable_request             N               Drill_Machine_not_initialized             N               Drilling                Generic_Device    N               Drilling_enable_request             N               Drilling_disable_request             N               Drilling_not_initialized             N                              Drilling_Handler                   Subsystem_Handler   N               enable_Drill_Machine            N %              disable_Drill_Machine            N &              Drill_Machine_enabled            N '              Drill_Machine_disabled            N (               Drill_Machine_EnabledSensorFault            N )           !   Drill_Machine_DisabledSensorFault            N *              Drill_Machine_fault            N +              Drill_Machine_ActuatorFault            N ,           Drilling - SA_NF    enable_Drilling            N /              Drilling_ActuatorFault            N 0           Holding - SA_NF    enable_Holding            N 3              Holding_ActuatorFault            N 4           Pure sensors *   AvailableLoadInDrillingPositioning_Logical            N 7              DrillingUnitDown_Logical            N 8              DrillingUnitUp_Logical            N 9               ��d  @   ����           GENERIC_DEVICE           DeviceState               GenericDevice_States    Q 4              DeviceTimer            Q 5              DeviceTimeout             Q 6              temp            Q 7              temp_int            Q 8          TImeout: BOOL; (* new variable   DeviceDisabled             Q ?              DeviceEnabled             Q @           	      DeviceOperation           Q            
   DeviceType           Q               DeviceEnabledSensor            Q               DeviceDisabledSensor            Q               DeviceClock            Q               DeviceEnableTime           Q 	              DeviceDisableTime           Q 
              DeviceDiagnosticsEnabled            Q               DeviceEnablePreset            Q                  EnableDevice            Q               DisableDevice            Q               DeviceEnabledSensorFault            Q               DeviceDisabledSensorFault            Q               DeviceActuatorFault            Q               DeviceFault            Q               DeviceNotInitialized            Q                  DeviceEnableRequest            Q               DeviceDisableRequest            Q                  INIT           Q                RUN          Q !              DEVICE_WITHOUT_FEEDBACK    @      Q &              DEVICE_WITH_ENABLE_FEEDBACK          Q '              DEVICE_WITH_DISABLE_FEEDBACK           Q (              DEVICE_WITH_DOUBLE_FEEDBACK    0      Q )              DEVICE_FEEDBACK_MASK    �      Q *       	   Actuation   DEVICE_WITH_SINGLE_ACTUATION          Q ,              DEVICE_WITH_DOUBLE_ACTUATION          Q -              DEVICE_WITH_DA_NO_RETAIN          Q .              DEVICE_ACTUATION_MASK          Q /               ��d  @    ����           INSPECTIONUNIT_PRG           OperationType            I               state           Inspection_ready_to_initialize       InspectionUnit_States    I 	              CylinderToInspect                                       Generic_Device    I                CylinderToInspect_enable_request             I            !   CylinderToInspect_disable_request             I            !   CylinderToInspect_not_initialized             I                              Inspection_Handler                   Subsystem_Handler   I               Inspection_SubData                   Data_Handler   I               enable_CylinderToInspect            I               CylinderToInspect_ActuatorFault            I            Pure Sensors %   AvailableLoadInControlPositionLogical            I !           .   InControlLoadInWrongPositionToBeDrilledLogical            I "           True se orientamento giusto    ��d  @   ����           ITEMSASSEMBLING_PRG           OperationType            a               state       (    ItemsAssemblingUnit_ready_to_initialize       ItemsAssemblingUnit_States    a               R_Initial_position                                       Generic_Device    a           R1!   R_Initial_position_enable_request             a            "   R_Initial_position_disable_request             a            "   R_Initial_position_not_initialized             a               R_TakeSpring                                       Generic_Device    a           R6   R_TakeSpring_enable_request             a               R_TakeSpring_disable_request             a               R_TakeSpring_not_initialized             a               R_TakePiston                                       Generic_Device    a           R7   R_TakePiston_enable_request             a               R_TakePiston_disable_request             a               R_TakePiston_not_initialized             a               R_TakeCover                                       Generic_Device    a           R8   R_TakeCover_enable_request             a               R_TakeCover_disable_request             a                R_TakeCover_not_initialized             a !              R_AssemblyPosition                                       Generic_Device    a #          R9!   R_AssemblyPosition_enable_request             a $           "   R_AssemblyPosition_disable_request             a %           "   R_AssemblyPosition_not_initialized             a &                             ItemsAssembling_Handler                   Subsystem_Handler   a 	              enable_R_Initial_position_A            a -              R_Initial_position_enabled_A            a .              R_Initial_position_fault_A            a /          R6   enable_R_TakeSpring            a 1              R_TakeSpring_enabled            a 2              R_TakeSpring_fault            a 3          R7   enable_R_TakePiston            a 6              R_TakePiston_enabled            a 7              R_TakePiston_fault            a 8          R8   enable_R_TakeCover            a ;              R_TakeCover_enabled            a <              R_TakeCover_fault            a =          R9   enable_R_AssemblyPosition            a @              R_AssemblyPosition_enabled            a A              R_AssemblyPosition_fault            a B               ��d  @   ����           ITEMSSUPPLY_PRG           OperationType            m               state       $    ItemsSupplyUnit_ready_to_initialize       ItemsSupplyUnit_States    m               PistonSelector                                       Generic_Device    m               PistonSelector_enable_request             m               PistonSelector_disable_request             m               PistonSelector_not_initialized             m               ExtractCover                                       Generic_Device    m               ExtractCover_enable_request             m               ExtractCover_disable_request             m               ExtractCover_not_initialized             m               ExtractSpring                                       Generic_Device    m               ExtractSpring_enable_request             m               ExtractSpring_disable_request             m               ExtractSpring_not_initialized             m                              ItemsSupply_Handler                   Subsystem_Handler   m 	              ItemsSupply_SubData                   Data_Handler   m 
              enable_PistonSelector            m #              disable_PistonSelector            m $              PistonSelector_enabled            m %              PistonSelector_disabled            m &           !   PistonSelector_EnabledSensorFault            m '           "   PistonSelector_DisabledSensorFault            m (              PistonSelector_fault            m )              PistonSelector_ActuatorFault            m *          ExtractCover - SA_DF   enable_ExtractCover            m -              ExtractCover_enabled            m .              ExtractCover_disabled            m /              ExtractCover_EnabledSensorFault            m 0               ExtractCover_DisabledSensorFault            m 1              ExtractCover_fault            m 2              ExtractCover_ActuatorFault            m 3          ExtractSpring - SA-DF   enable_ExtractSpring            m 6              ExtractSpring_enabled            m 7              ExtractSpring_disabled            m 8               ExtractSpring_EnabledSensorFault            m 9           !   ExtractSpring_DisabledSensorFault            m :              ExtractSpring_fault            m ;              ExtractSpring_ActuatorFault            m <               ��d  @   ����           LIGHTS                           ResetSignalsEnable            %               LightEmptyWarehouseLogical            % 	              LightEmptyCoverhouseLogical            % 
          in assembly station   LightStartLogical            %            
   LightStart            %            
   LightReset            %               LightEmptyWarehouse            %               LightEmptyCoverhouse            %           in assembly station    ��d  @    ����           MAIN_PRG           state           Ready_to_initialize       Main_States    \                              Init_Logical            \               Start_Logical            \ 	              Reset_Logical            \ 
              Stop_Logical            \               OnPhaseStop_Logical            \               ImmediateStop_Logical            \               DistTest_Handler                 System_Handler   \               Making_Handler                 System_Handler   \               Robot_Handler                 System_Handler   \               SignalManagement_Handler                 System_Handler   \ "              LightStartLogical            \ '               ��d  @    ����        
   MAKING_PRG           state_Making       !    M_Processing_ready_to_initialize       Making_States    L 7                             Memory_Data   	                        Data_Handler           L               Distribution_index         L 	              Testing_index         L 
              Rotary_index         L               Inspection_index         L               Drilling_index         L               Expelling_index         L               PickandPlace_index         L               Supply_index         L               Making_Handler                 System_Handler   L               Processing_Handler                   Subsystem_Handler   L               Rotary_Data                   Data_Handler   L               Inspection_Data                   Data_Handler   L               Drilling_Data                   Data_Handler   L               Expelling_Data                   Data_Handler   L                Testing_ready_to_send            L %              Processing_ready_to_receive            L *              Robot_ready_to_receive           L /               ��d  @    ����           PICKANDPLACE_PRG           OperationType            (               state       !    PickandPlace_ready_to_initialize       PickandPlace_States    (               R_Initial_position                                       Generic_Device    (            !   R_Initial_position_enable_request             (            "   R_Initial_position_disable_request             (            "   R_Initial_position_not_initialized             (           R2   R_Take_black_piece                                       Generic_Device    (            !   R_Take_black_piece_enable_request             (            "   R_Take_black_piece_disable_request             (            "   R_Take_black_piece_not_initialized             (           R3   R_Take_redsilver_piece                                       Generic_Device    (            %   R_Take_redsilver_piece_enable_request             (             &   R_Take_redsilver_piece_disable_request             ( !           &   R_Take_redsilver_piece_not_initialized             ( "          R4   R_Take_black_upsidedown_piece                                       Generic_Device    ( %           ,   R_Take_black_upsidedown_piece_enable_request             ( &           -   R_Take_black_upsidedown_piece_disable_request             ( '           -   R_Take_black_upsidedown_piece_not_initialized             ( (          R5!   R_Take_redsilver_upsidedown_piece                                       Generic_Device    ( +           0   R_Take_redsilver_upsidedown_piece_enable_request             ( ,           1   R_Take_redsilver_upsidedown_piece_disable_request             ( -           1   R_Take_redsilver_upsidedown_piece_not_initialized             ( .                             PickandPlace_Handler                   Subsystem_Handler   (               PickandPlace_Data                   Data_Handler   (               enable_R_Initial_position            ( 5              R_Initial_position_enabled            ( 6              R_Initial_position_fault            ( 7          R2   enable_R_Take_black_piece            ( :              R_Take_black_piece_enabled            ( ;              R_Take_black_piece_fault            ( <          R3   enable_R_Take_redsilver_piece            ( ?              R_Take_redsilver_piece_enabled            ( @              R_Take_redsilver_piece_fault            ( A          R4$   enable_R_Take_black_upsidedown_piece            ( D           %   R_Take_black_upsidedown_piece_enabled            ( E           #   R_Take_black_upsidedown_piece_fault            ( F          R5(   enable_R_Take_redsilver_upsidedown_piece            ( I           )   R_Take_redsilver_upsidedown_piece_enabled            ( J           '   R_Take_redsilver_upsidedown_piece_fault            ( K           Pure sensors    AvailableLoadForRobot_Logical_PP            ( N               ��d  @   ����           PLANTASSEMBLAGGIO           State            �              Count             �              Count2             �           	   CaseBlack            �              CaseRedSilver            �              CaseOverturned            �              InitialPosition            �	           -   FLAGToExtractSpringInAssemblyStationBlockHigh             �
           5   FLAGBlockingCylinderForwardInAssemblyStationBlockHigh             �           3   FLAGToExtractCoverInAssemblyStationForwardBlockHigh             �              Count3             �              Count4             �                               ��d  @    ����           PLANTCARICO           EstrazioneRosso            1        =   variabile che memorizza la posizione dei pezzi rossi estratti   EstrazioneSilver            1        A   variabile che memorizza la posizione dei pezzi argentati estratti   EstrazioneNero            1        <   variabile che memorizza la posizione dei pezzi neri estratti                    ��d  @    ����           PLANTLAVORAZIONE     	      Set            �              FLAG            �              ElementInControlOverturned             �              FLAGRotaryTableMotorBlockHigh             �           )   FLAGToLowerCylinderToInspectLoadBlockHigh             �           9   FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh             �               FLAGToLowerDrillingUnitBlockHigh             �	              FLAGToLiftDrillingUnitBlockHigh             �
           !   FLAGExpellingLeverActiveBlockHigh             �                               ��d  @    ����           PLANTMAGAZZINO           BlockedVsVerification             2        H   flag che segnala il bloccaggio del braccio rotante in posizione verifica   BlockedVsWarehouse             2        I   flag che segnala il bloccaggio del braccio rotante in posizione magazzino                    ��d  @    ����           PLANTSCARICO                             ��d  @    ����           PLANTVERIFICATION        	   NoElement             5        |   variabile che segnala se il pezzo � gia stato preso dal rotary maker (necessaria nel caso di estrattore-basi bloccato alto)    BlockedHigh             5        2   flag che segnala il bloccaggio alto dell'ascensore
   BlockedLow             5        3   flag che segnala il bloccaggio basso dell'ascensore                    ��d  @    ����           PROCESSING_PRG           OperationType    �      D        -    To menage the init phase of Generic Devices    state           Processing_ready_to_initialize       Processing_States    D            	   Null_Data                   Data_Handler    D               RotaryTable                                       Generic_Device    D               RotaryTable_enable_request             D                RotaryTable_disable_request             D !              RotaryTable_not_initialized             D "              ExpellingLever                                       Generic_Device    D $              ExpellingLever_enable_request             D %              ExpellingLever_disable_request             D &              ExpellingLever_not_initialized             D '                             Processing_Handler                   Subsystem_Handler   D               Rotary_Data                   Data_Handler   D               Inspection_Data                   Data_Handler   D               Drilling_Data                   Data_Handler   D               Expelling_Data                   Data_Handler   D               Inspection_Handler                   Subsystem_Handler   D               Inspection_SubData                   Data_Handler   D               Drilling_Handler                   Subsystem_Handler   D               enable_RotaryTable            D -              disable_RotaryTable            D .              RotaryTable_enabled            D /              RotaryTable_EnabledSensorFault            D 0              RotaryTable_disabled            D 1              RotaryTable_DisabledSensorFault            D 2              RotaryTable_fault            D 3              RotaryTable_ActuatorFault            D 4           ExpellingLever - SA_NF    enable_ExpellingLever            D 7              ExpellingLever_ActuatorFault            D 8           Pure sensors &   AvailableLoadForWorkingStation_Logical            D ;               ��d  @    ����           PULSANTIERA                             ��d  @    ����        	   ROBOT_PRG           state_Robot           Robot_ready_to_initialize       Robot_States    + .           	   Null_Data                   Data_Handler    + 1                             Memory_Data   	                        Data_Handler           +               Distribution_index         + 	              Testing_index         + 
              Rotary_index         +               Inspection_index         +               Drilling_index         +               Expelling_index         +               PickandPlace_index         +               Supply_index         +               Robot_Handler                 System_Handler   +               PickandPlace_Handler                   Subsystem_Handler   +               PickandPlace_Data                   Data_Handler   +               Assembly_Handler                   Subsystem_Handler   + #              Assembly_Data                   Data_Handler   + $              Robot_ready_to_receive           + )               ��d  @    ����        	   SAVE_DATA        	   Index7001                            Memory_Index           e               Memory_Data   	                        Data_Handler           e               To_Save_Data                   Data_Handler   e               	   Save_data   	                        Data_Handler                                     ��d  @    ����        
   SHIFT_DATA        
   Empty_Data                   Data_Handler    d 	           	   Index7001                            Memory_Index           d               Memory_Data   	                        Data_Handler           d               
   Shift_data   	                        Data_Handler                                     ��d  @    ����           SIGNAL_FILTER           Delay            ^                  Signal            ^               ActivationDelay          ^               DeactivationDelay          ^                  DelayedSignal            ^ 	                       ��d  @    ����           SIGNALCONTROL_PRG     &      SignalManagement                SignalManagement    Y               OutputSignals            Y               ResetEnable             Y               mCylinder_EnabledSensorFault          Y               mCylinder_DisabledSensorFault          Y               mCylinder_ActuatorFault          Y               mRotaryMaker_EnabledSensorFault          Y                mRotaryMaker_DisabledSensorFault          Y               mRotaryMaker_ActuatorFault          Y            #   mVacuumGenerator_EnabledSensorFault          Y               mVacuumGenerator_ActuatorFault          Y               mElevator_EnabledSensorFault    	      Y               mElevator_DisabledSensorFault    
      Y               mElevator_ActuatorFault          Y            '   mExtractionCylinder_DisabledSensorFault          Y            !   mExtractionCylinder_ActuatorFault          Y               mRotaryTable_EnabledSensorFault          Y                mRotaryTable_DisabledSensorFault          Y               mRotaryTable_ActuatorFault          Y            !   mDrill_Machine_EnabledSensorFault          Y            "   mDrill_Machine_DisabledSensorFault          Y               mDrill_Machine_ActuatorFault          Y            "   mPistonSelector_EnabledSensorFault          Y            #   mPistonSelector_DisabledSensorFault          Y                mPistonSelector_ActuatorFault          Y !               mExtractCover_EnabledSensorFault          Y "           !   mExtractCover_DisabledSensorFault          Y #              mExtractCover_ActuatorFault          Y $           !   mExtractSpring_EnabledSensorFault          Y %           "   mExtractSpring_DisabledSensorFault          Y &              mExtractSpring_ActuatorFault          Y '              mEmptyWarehouse_Logical          Y (           )   mEmptyCoverHouseInAssemblyStation_Logical          Y )              EMERGENCY_STOP          Y .              IMMEDIATE_STOP          Y /              ON_PHASE_STOP          Y 0              LIGHT_EMPTYWAREHOUSE          Y u              LIGHT_EMPTYCOVERHOUSE          Y v                             SignalManagement_Handler                 System_Handler   Y 7              Reset_Logical            Y <              Cylinder_EnabledSensorFault            Y C              Cylinder_DisabledSensorFault            Y D              Cylinder_fault            Y E              Cylinder_ActuatorFault            Y F           RotaryMaker - DA_DF    RotaryMaker_EnabledSensorFault            Y I              RotaryMaker_DisabledSensorFault            Y J              RotaryMaker_fault            Y K              RotaryMaker_ActuatorFault            Y L           VacuumGenerator - SA_SAF "   VacuumGenerator_EnabledSensorFault            Y O              VacuumGenerator_fault            Y P              VacuumGenerator_ActuatorFault            Y Q       #    GDs - Sensors - Logical variables     ExpulsionAirVacuum_ActuatorFault            Y T              ResetSignalsEnable            Y Z              ImmediateStop_Signal            Y \              OnPhaseStop_Signal            Y ]              SIGNAL_TYPE_MASK         Y e          ultimi 3 bit   ALARM         Y f              ANOMALY         Y g              WARNING         Y h              INFORMATION         Y i           To clean the variables    NONE          Y l       �   Dimensione massima dell'arrey contenente i segnali.
    Ovvero il numero massimo del "SignalCode" da associare ai segnali, generati dal file Python   N          Y p              LightEmptyWarehouseLogical            Y z              LightEmptyCoverhouseLogical            Y {          in assembly station    ��d  @    ����           SIGNALMANAGEMENT           NumberOfAlarms            o               NumberOfAnomalies            o               NumberOfWarnings            o               NumberOfInformation            o            START_GENERATION    ResetOld             o        ?   Per mantenere la memoria del segnale e effetuare il risign edge   ResetActivation             o        "    Rising edge del segnale di reset    KeyResetOld             o !       ?   Per mantenere la memoria del segnale e effetuare il risign edge   AuxResetActivation             o "       %    Rising edge del segnale di keyreset    Index            o %              Current_index            o &       *   Auxiliary index for auto-conditioned reset	   BaseIndex            o '              NumberOfActiveAlarms            o (              NumberOfActiveAnomalies            o )              NumberOfActiveWarning            o *              NumberOfActiveInformation            o +           	   Condition             o ,       L    Serve per differenziare fra Allarms o Information, i primi vanno resettati    Signals   	                           o /       M    Insieme di tutti i segnali, ove il suo indice � il SignalCode dato in input    ActiveSignalCodes   	                          o 0       A    Salviamo in ordine di attivazione i codici dei segnali attivati    i           o 1       1    Usato come puntatore in un ciclo FOR nel codice       OperationType_States           INIT_SIGNALMANAGEMENT       SignalManagement_States   o           States
   SignalType           o        `    Usato come Array di dimensione 16, i cui primi 3 bit rappresentano i possibili tipi di segnale 
   SignalCode           o        M    Codice del segnale di ingresso viene usato come indice nell'Array 'Signals'    SignalOutput           o        �    Array di bit il cui ogni bit identifica un reazione al dato segnale. Verr� usato per settare il relativo bit nella variabile di 'SignalOutputs'    ActivationSignal            o            Segnale di ingresso     AutoResetSignal            o            ???    Reset            o        -    Possibile segnale di input dal bridge fisico   KeyReset            o        -    Possibile segnale di input dal bridge fisico      ResetEnable            o        P   Richiesta di reset. Accensione del led di reset, � possibile effetuare il reset    SignalOutputs           o        M    Un Array di dimensione 32, i cui primi 3 bit rappresentano i possibili STOP     
      SIGNAL_TYPE_MASK         o 8          ultimi 3 bit   ALARM         o 9              ANOMALY         o :              WARNING         o ;              INFORMATION         o <           Signal Reset Definitions    SIGNAL_RESET_MASK     <    o ?          10 A 13 bit
   AUTO_RESET         o @              AUTO_CONDITIONED_RESET          o A           To clean the variables    NONE          o D       �   Dimensione massima dell'arrey contenente i segnali.
    Ovvero il numero massimo del "SignalCode" da associare ai segnali, generati dal file Python   N          o H               ��d  @   ����           TESTING_COLOUR               Color            i               Height            i                  Testing_colour                                      ��d  @    ����           TESTING_ORIENTATION               Orientation            :                  Testing_orientation                                      ��d  @    ����           TESTING_PRG           OperationType            X               state           Testing_ready_to_initialize       Testing_States    X               Elevator                                       Generic_Device    X               Elevator_enable_request             X               Elevator_disable_request             X               Elevator_not_initialized             X               ExtractionCylinder                                       Generic_Device    X            !   ExtractionCylinder_enable_request             X            "   ExtractionCylinder_disable_request             X            "   ExtractionCylinder_not_initialized             X            
   AirCushion                                       Generic_Device    X               AirCushion_enable_request             X               AirCushion_disable_request             X               AirCushion_not_initialized             X                               Testing_Handler                   Subsystem_Handler   X               Testing_Data                   Data_Handler   X               enable_Elevator            X &              disable_Elevator            X '              Elevator_enabled            X (              Elevator_disabled            X )              Elevator_EnabledSensorFault            X *              Elevator_DisabledSensorFault            X +              Elevator_fault            X ,              Elevator_ActuatorFault            X -          ExtractionCylinder - SA_SDF    enable_ExtractionCylinder            X 0              ExtractionCylinder_disabled            X 1           &   ExtractionCylinder_DisabledSensorFault            X 2              ExtractionCylinder_fault            X 3               ExtractionCylinder_ActuatorFault            X 4          AirCushion - SA_NF   enable_AirCushion            X 7              AirCushion_ActuatorFault            X 8           Pure sensors    ReadyLoadForVerificationLogical            X ;              MeasurementNotOkLogical            X <              ColourMeasurementLogical            X =               ��d  @    ����            
 �   /   V   ( �      K   ��     K   ��     K   ��     K   ��                 ˹         +     ��localhost ys V2.3\CoDeSys exe Y��                                       �Y  	Y����   �Y@   ���      ��Y     � ��� H+� � /R� �� 3��  �C�      ��       4   ��       L� ��� H+� \�  �� 	    �Cx� ��                   �C̤     ,   ,                                                        K        @   ��d��      ,   ��                     CoDeSys 1-2.2   ����  ��������                                �      
   �         �         �          �                    "          $                                                   '          (          �          �          �          �          �         �          �          �          �         �          �          �          �          �         �      �   �       P  �          �         �       �  �                    ~          �          �          �          �          �          �          �          �          �          �          �          �          �          �          �          �          �       @  �       @  �       @  �       @  �       @  �       @  �         �         �          �       �  M         N          O          P          `         a          t          y          z          b         c          X          d         e         _          Q          \         R          K          U         X         Z         �          �         �      
   �         �         �         �         �         �          �          �         �      �����          �          �      (                                                                        "         !          #          $         �          ^          f         g          h          i          j          k         F          H         J         L          N         P         R          U         S          T          V          W          �          �          l          o          p          q          r          s         u          �          v         �          �      ����|         ~         �         x          z      (   �          �         %         �          �          �         @         �          �          �         &          �          	                   �          �          �         �          �         �          �          �          �          �          �          �          �          �          �          �          �                            I         J         K          	          L         M          �                             �          P         Q          S          )          	          	          �           	          +	       @  ,	       @  -	      ����Z	      ����[	      ��������        ������������  ��������                                                   �  	   	   Name                 ����
   Index                 ��         SubIndex                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write       Variable    	             ����
   Value                Variable       Min                Variable       Max                Variable          5  
   	   Name                 ����
   Index                 ��         SubIndex                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write    	   Type          ~         INT   UINT   DINT   UDINT   LINT   ULINT   SINT   USINT   BYTE   WORD   DWORD   REAL   LREAL   STRING    
   Value                Type       Default                Type       Min                Type       Max                Type          5  
   	   Name                 ����
   Index                 ��         SubIndex                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write    	   Type          ~         INT   UINT   DINT   UDINT   LINT   ULINT   SINT   USINT   BYTE   WORD   DWORD   REAL   LREAL   STRING    
   Value                Type       Default                Type       Min                Type       Max                Type          d        Member    	             ����   Index-Offset                 ��         SubIndex-Offset                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write       Min                Member       Max                Member          �  	   	   Name                 ����   Member    	             ����
   Value                Member    
   Index                 ��         SubIndex                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write       Min                Member       Max                Member          �  	   	   Name                 ����
   Index                 ��         SubIndex                 �          Accesslevel          !         low   middle   high       Accessright          1      	   read only
   write only
   read-write       Variable    	             ����
   Value                Variable       Min                Variable       Max                Variable                         ����  ��������               �   _Dummy@    @   @@    @   @             ��@             ��@@   @     �v@@   ; @+   ����  ��������                                  �v@      4@   �             �v@      D@   �                       �       @                           �f@      4@     �f@                �v@     �f@     @u@     �f@        ���           __not_found__-1__not_found__    __not_found__     IB          % QB          % MB          %    ��d	��d     ��������           VAR_GLOBAL
END_VAR
                                                                                  "   , � ��              
Simulation
        PlantVerification();PlantAssemblaggio();PlantLavorazione();PlantCarico();PlantMagazzino();PlantScarico();Pulsantiera();����                 Bridges
       	 Bridge_Processing_GDs();Bridge_Inspection_GSs();Bridge_Drilling_GDs();Bridge_Testing_GDs();Bridge_Distribution_GDs();Bridge_PickandPlace_GDs();Bridge_Assembly_GDs();Bridge_ItemsSupply_GDs();Bridge_ItemsAssembling_GDs();����                 
Subsystems
       	 InspectionUnit_PRG();DrillingUnit_PRG();Processing_PRG();Distribution_PRG();Testing_PRG();PickandPlace_PRG();Assembly_PRG();ItemsSupply_PRG();ItemsAssembling_PRG();����                 Systems
        DistTest_PRG();Main_PRG();Making_PRG();Robot_PRG();����                InputBridge_filter
        
Buttons();	Lights();����                SignalManagement         SignalControl_PRG();����               ��d                 $����, , : @X               ��������           dani S��d	S��d      ��������                  ��������           Watch0 S��d	S��d      ��������        N   TEST_GD.cylinder.DeviceDiagnosticsEnabled

TEST_GD.cylinder_enable_request
             	s��d     ��������           VAR_CONFIG
END_VAR
                                                                                   '           <   ,  �           Assembly_Global_Variables ��d	��d<     ��������        �   (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(*BlockingCylinder - SA_NF*)
     enable_BlockingCylinder : BOOL;

	(* Pure sensors *)
	EmptyCoverHouseInAssemblyStation_Logical: BOOL;
END_VAR
                                                                                               '           W   ,  K           Button_Bridge_Global_Variables ��d	��dW     ��������        p  (*Output - Logical*)
VAR_GLOBAL
	Init_Logical : BOOL;
	Start_Logical : BOOL;
	Reset_Logical : BOOL;
	Stop_Logical : BOOL;
	OnPhaseStop_Logical: BOOL;
	ImmediateStop_Logical:BOOL;
END_VAR

(*Input -HMI Panel*)
VAR_GLOBAL
	Init_HMI : BOOL;
	Start_HMI : BOOL;
	Stop_HMI : BOOL;
	Reset_HMI : BOOL;
	OnPhaseStop_HMI: BOOL;
	ImmediateStop_HMI:BOOL;
END_VAR                                                                                               '           G   , ���           Distribution_Global_Variables ��d	��dG     ��������        u  (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(* Cylinder - SA_DF *)
     enable_Cylinder : BOOL;
     Cylinder_enabled : BOOL;
     Cylinder_disabled : BOOL;

	(* RotaryMaker - DA_DF *)
     enable_RotaryMaker : BOOL;
     disable_RotaryMaker : BOOL;
     RotaryMaker_enabled : BOOL;
     RotaryMaker_disabled : BOOL;

	(* VacuumGenerator - SA_SAF *)
     enable_VacuumGenerator : BOOL;
     VacuumGenerator_enabled : BOOL;

	(* ExpulsionAirVacuum - SA_NF *)
     enable_ExpulsionAirVacuum : BOOL;

END_VAR

(* GDs - Sensors - Logical variables *)
VAR_GLOBAL
	EmptyWarehouse_Logical:BOOL:=TRUE;
END_VAR
                                                                                               '           P   ,  ��           DrillingUnit_Global_Variables ��d	��dP     ��������        �  
(* GDs - Actuators and Sensors*)
VAR_GLOBAL
   (* Drill_Machine - DA_DF *)
     enable_Drill_Machine : BOOL;
     disable_Drill_Machine : BOOL;
     Drill_Machine_enabled : BOOL;
     Drill_Machine_disabled : BOOL;

   (* Drilling - SA_NF *)
     enable_Drilling : BOOL;

  (* Holding - SA_NF *)
     enable_Holding : BOOL;

   (* Pure sensors *)
   AvailableLoadInDrillingPositioning_Logical : BOOL;
   DrillingUnitDown_Logical : BOOL;
   DrillingUnitUp_Logical : BOOL;
END_VAR
                                                                                               '           b   , eQ �q           GDs_Timeout_Configuration ��d	��db     ��������        �  (*In queste config mancano quelle del robot, quindi delle stazioni PickandPlace e ItemsAssembling*)

VAR_GLOBAL CONSTANT
	(*Distribution*)
     Cylinder_EnableTime : INT :=30;
     Cylinder_DisableTime : INT :=30;
     RotaryMaker_EnableTime : INT :=100;
     RotaryMaker_DisableTime : INT :=100;
     VacuumGenerator_EnableTime : INT :=100;
     VacuumGenerator_DisableTime : INT :=10;
     ExpulsionAirVacuum_EnableTime : INT :=10;
     ExpulsionAirVacuum_DisableTime : INT :=10;

	(*Testing*)
     Elevator_EnableTime : INT :=100;
     Elevator_DisableTime : INT :=100;
     ExtractionCylinder_EnableTime : INT :=30;
     ExtractionCylinder_DisableTime : INT :=30;
     AirCushion_EnableTime : INT :=30;
     AirCushion_DisableTime : INT :=5;

	(*Processing*)
     RotaryTable_EnableTime : INT :=200;
     RotaryTable_DisableTime : INT :=200;
     ExpellingLever_EnableTime : INT :=20;
     ExpellingLever_DisableTime : INT :=20;

	(*Inspection Unit*)
     CylinderToInspect_EnableTime : INT :=30;
     CylinderToInspect_DisableTime : INT :=30;

	(*Drilling Unit*)
     Drill_Machine_EnableTime : INT :=100;
     Drill_Machine_DisableTime : INT :=100;
     Drilling_EnableTime : INT :=30;
     Drilling_DisableTime : INT :=30;
     Holding_EnableTime : INT :=30;
     Holding_DisableTime : INT :=30;

	(*Assembly*)
     BlockingCylinder_EnableTime : INT :=20;
     BlockingCylinder_DisableTime : INT :=20;

	(*ItemsSupply*)
     PistonSelector_EnableTime : INT :=30;
     PistonSelector_DisableTime : INT :=30;
     ExtractCover_EnableTime : INT :=30;
     ExtractCover_DisableTime : INT :=30;
     ExtractSpring_EnableTime : INT :=30;
     ExtractSpring_DisableTime : INT :=30;
END_VAR
                                                                                               '           S   , SS ��           GenericDevice_Global_Variables ��d	��dS     ��������        l  VAR_GLOBAL CONSTANT
	INIT :	INT:=0;
	RUN :	INT:=1;
END_VAR

(*Feedback*)
VAR_GLOBAL CONSTANT
	DEVICE_WITHOUT_FEEDBACK : 		BYTE := 2#01000000;
	DEVICE_WITH_ENABLE_FEEDBACK : 	BYTE := 2#00010000;
	DEVICE_WITH_DISABLE_FEEDBACK : 	BYTE := 2#00100000;
	DEVICE_WITH_DOUBLE_FEEDBACK : 	BYTE := 2#00110000;
	DEVICE_FEEDBACK_MASK : 			BYTE := 2#11110000;
END_VAR

(*Actuation*)
VAR_GLOBAL CONSTANT
	DEVICE_WITH_SINGLE_ACTUATION : 	BYTE := 2#00000001;
	DEVICE_WITH_DOUBLE_ACTUATION : 	BYTE := 2#00000011;
	DEVICE_WITH_DA_NO_RETAIN : 		BYTE := 2#00000010;
	DEVICE_ACTUATION_MASK : 			BYTE := 2#00001111;
END_VAR                                                                                               '           {   , $ v�        &   Handlers_Comunication_Global_Variables ��d	��d{     ��������        _	  (******* From *** MAIN *********)
(* Between MAIN_Prg and DISTTEST_Prg *)
VAR_GLOBAL
	DistTest_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and MAKING_Prg *)
VAR_GLOBAL
	Making_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and ROBOT_Prg *)
VAR_GLOBAL
	Robot_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and SIGNALMANAGEMENT_Prg *)
VAR_GLOBAL
	SignalManagement_Handler : 	System_Handler;
END_VAR



(******* From *** DISTTEST *********)
(* Between DISTTEST_Prg and DISTRIBUTION_Prg *)
VAR_GLOBAL
	Distribution_Handler:		Subsystem_Handler;
	Distribution_Data:		Data_Handler;
END_VAR

(* Between DISTTEST_Prg and TESTING_Prg *)
VAR_GLOBAL
	Testing_Handler:		Subsystem_Handler;
	Testing_Data:		Data_Handler;
END_VAR



(******* From *** MAKING *********)
(* Between MAKING_Prg and PROCESSING_Prg *)
VAR_GLOBAL
	Processing_Handler:		Subsystem_Handler;
	Rotary_Data:			Data_Handler;
	Inspection_Data:			Data_Handler;
	Drilling_Data:			Data_Handler;
	Expelling_Data:			Data_Handler;
END_VAR

(******* From *** PROCESSING *********)
(* Between PROCESSING_Prg and INSPECTION_UNIT_Prg *)
VAR_GLOBAL
	Inspection_Handler:	Subsystem_Handler;
	Inspection_SubData:	Data_Handler;
END_VAR

(* Between PROCESSING_Prg and DRILLING_UNIT_Prg *)
VAR_GLOBAL
	Drilling_Handler:	Subsystem_Handler;
END_VAR




(******* From *** ROBOT *********)
(* Between ROBOT_Prg and PICKANDPLACE_Prg*)
VAR_GLOBAL
	PickandPlace_Handler : 	Subsystem_Handler;
	PickandPlace_Data :		Data_Handler;
END_VAR

(* Between ROBOT_Prg and ASSEMBLY_PRG *)
VAR_GLOBAL
	Assembly_Handler:		Subsystem_Handler;
	Assembly_Data:			Data_Handler;
END_VAR

(******* From *** ASSEMBLY *********)
(* Between ASSEMBLY_Prg and ITEMSSUPPLY_Prg*)
VAR_GLOBAL
	ItemsSupply_Handler : 		Subsystem_Handler;
	ItemsSupply_SubData:		Data_Handler;
END_VAR

(* Between ASSEMBLY_Prg and ITEMSASSEMBLING_Prg*)
VAR_GLOBAL
	ItemsAssembling_Handler : 		Subsystem_Handler;
END_VAR



(******* Between *** MACHINE SYSTEMS *********)
(* Between DISTTEST_Prg and MAKING_Prg *)
VAR_GLOBAL
	Testing_ready_to_send :BOOL:= FALSE;
END_VAR

(* Between MAKING_Prg and DISTTEST_Prg *)
VAR_GLOBAL
      Processing_ready_to_receive :BOOL:= FALSE;
END_VAR

(* Between ASSEMBLY_Prg and MAKING_Prg *)
VAR_GLOBAL
	Robot_ready_to_receive:BOOL:= TRUE;
END_VAR


                                                                                               '           J   , 	 �           InspectionUnit_Global_Variables ��d	��dJ     ��������        1  (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(* CylinderToInspect - SA_NF *)
     enable_CylinderToInspect : BOOL;

	(* Pure Sensors *)
	AvailableLoadInControlPositionLogical : BOOL := FALSE;
	InControlLoadInWrongPositionToBeDrilledLogical : BOOL := FALSE; (* True se orientamento giusto*)
END_VAR                                                                                               '           c   ,  + W$            ItemsAssembling_Global_Variables ��d	��dc     ��������        �  VAR_GLOBAL
   (* GDs- Robot - SA-SAF*)
	(*R1*)
   enable_R_Initial_position_A : BOOL;
   R_Initial_position_enabled_A : BOOL;
   R_Initial_position_fault_A :BOOL;
	(*R6*)
   enable_R_TakeSpring : BOOL;
   R_TakeSpring_enabled : BOOL;
   R_TakeSpring_fault :BOOL;

	(*R7*)
   enable_R_TakePiston : BOOL;
   R_TakePiston_enabled : BOOL;
   R_TakePiston_fault :BOOL;

	(*R8*)
   enable_R_TakeCover : BOOL;
   R_TakeCover_enabled : BOOL;
   R_TakeCover_fault :BOOL;

	(*R9*)
   enable_R_AssemblyPosition : BOOL;
   R_AssemblyPosition_enabled : BOOL;
   R_AssemblyPosition_fault :BOOL;
END_VAR


(*Logical variables to bridge the previous variables with the phisical actuators and feedback, bacause they are Bits*)
VAR_GLOBAL
	(* Bits to Actuate *)
	RobotTakeCurrentLoadToAssembly_Logical: BOOL;
	RobotGoToPistonHouse_Logical: BOOL;
	RobotGoToSpringHouse_Logical: BOOL;
	RobotGoToCoverHouse_Logical: BOOL;
	RobotGoToInitialPosition_Logical_A : BOOL;

	(* Feedback Bits *)
	RobotInAssemblyUnit_Logical_A: BOOL;
	RobotInPistonWarehouse_Logical: BOOL;
	RobotInSpringWarehouse_Logical :BOOL;
	RobotInCoverWarehouse_Logical :BOOL;
	RobotInInitialPosition_Logical_A : BOOL;
END_VAR                                                                                               '           k   , > �           ItemsSupply_Global_Variables ��d	��dk     ��������        �  (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(*PistonSelector - DA-DF*)
     enable_PistonSelector : BOOL;
     disable_PistonSelector : BOOL;
     PistonSelector_enabled : BOOL;
     PistonSelector_disabled : BOOL;

	(*ExtractCover - SA_DF*)
     enable_ExtractCover : BOOL;
     ExtractCover_enabled : BOOL;
     ExtractCover_disabled : BOOL;

	(*ExtractSpring - SA-DF*)
     enable_ExtractSpring : BOOL;
     ExtractSpring_enabled : BOOL;
     ExtractSpring_disabled : BOOL;
END_VAR
                                                                                               '           H   , � � ��           Light_Bridge_Global_Variables ��d	��dH     ��������        �   VAR_GLOBAL
	LightEmptyWarehouseLogical: BOOL;
	LightEmptyCoverhouseLogical: BOOL; (*in assembly station*)

	LightStartLogical: BOOL;
END_VAR
                                                                                               '           h   , +� ��           Memory_Global_Variables ��d	��dh     ��������        �  (* Memory array to manage data memory and its operation *)
VAR_GLOBAL
    	Memory_Data: ARRAY [1..8] OF Data_Handler;
END_VAR

(* Arrey's index *)
VAR_GLOBAL CONSTANT
	Distribution_index:		UINT:=1;
	Testing_index:			UINT:=2;
	Rotary_index:			UINT:=3;
	Inspection_index:			UINT:=4;
	Drilling_index:			UINT:=5;
	Expelling_index:			UINT:=6;
	PickandPlace_index:		UINT:=7;
	Supply_index:			UINT:=8;
END_VAR                                                                                               '           *   , 	  �#           PickandPlace_Global_Variables ��d	��d*     ��������        B  VAR_GLOBAL
(* GDs- Robot - SA-SAF*)
	(*R1*)
   enable_R_Initial_position : BOOL;
   R_Initial_position_enabled : BOOL;
   R_Initial_position_fault :BOOL;

	(*R2*)
   enable_R_Take_black_piece : BOOL;
   R_Take_black_piece_enabled : BOOL;
   R_Take_black_piece_fault :BOOL;

	(*R3*)
   enable_R_Take_redsilver_piece : BOOL;
   R_Take_redsilver_piece_enabled : BOOL;
   R_Take_redsilver_piece_fault :BOOL;

	(*R4*)
   enable_R_Take_black_upsidedown_piece : BOOL;
   R_Take_black_upsidedown_piece_enabled : BOOL;
   R_Take_black_upsidedown_piece_fault :BOOL;

	(*R5*)
   enable_R_Take_redsilver_upsidedown_piece : BOOL;
   R_Take_redsilver_upsidedown_piece_enabled : BOOL;
   R_Take_redsilver_upsidedown_piece_fault :BOOL;

	(* Pure sensors*)
	AvailableLoadForRobot_Logical_PP : BOOL;
END_VAR

(*Logical variables to bridge the previous variables with the phisical actuators and feedback, bacause they are Bits*)
VAR_GLOBAL
	(* Bits to Actuate *)
	RobotGoToInitialPosition_Logical : BOOL;
	RobotTakeBlackLoad_Logical : BOOL;
	RobotTakeRedSilverLoad_Logical : BOOL;
	RobotTakeLoadToDiascardBlack_Logical : BOOL;
	RobotTakeLoadToDiascardRedSilver_Logical : BOOL;

	RobotNullCommand_Logical:BOOL;

	(* Feedback Bits *)
	RobotInInitialPosition_Logical : BOOL;
	RobotInAssemblyUnit_Logical : BOOL;

END_VAR
                                                                                               '           ]   ,  ��           Processing_Global_Variables ��d	��d]     ��������        h  (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(* RotaryTable - DANR_DF *)
     enable_RotaryTable : BOOL;
     disable_RotaryTable : BOOL;
     RotaryTable_enabled : BOOL;
     RotaryTable_disabled : BOOL;

	(* ExpellingLever - SA_NF *)
     enable_ExpellingLever : BOOL;

	(* Pure sensors *)
	AvailableLoadForWorkingStation_Logical : BOOL;
END_VAR                                                                                               '           v   , U��g�           SignalControl_Global_Variables ��d	��dv     ��������        
  (*Fault from GD*)
VAR_GLOBAL

	(* DISTRIBUTION *)

	(* Cylinder - SA_DF *)
	Cylinder_EnabledSensorFault : BOOL;
	Cylinder_DisabledSensorFault : BOOL;
	Cylinder_fault :BOOL;
	Cylinder_ActuatorFault : BOOL;

	(* RotaryMaker - DA_DF *)
	RotaryMaker_EnabledSensorFault : BOOL;
	RotaryMaker_DisabledSensorFault : BOOL;
	RotaryMaker_fault :BOOL;
	RotaryMaker_ActuatorFault : BOOL;

	(* VacuumGenerator - SA_SAF *)
	VacuumGenerator_EnabledSensorFault : BOOL;
	VacuumGenerator_fault :BOOL;
	VacuumGenerator_ActuatorFault : BOOL;

	(* ExpulsionAirVacuum - SA_NF *)
	ExpulsionAirVacuum_ActuatorFault : BOOL;



	(* TESTING *)

	(*Elevator - DA_DF *)
	Elevator_EnabledSensorFault : BOOL;
	Elevator_DisabledSensorFault : BOOL;
	Elevator_fault :BOOL;
	Elevator_ActuatorFault : BOOL;

	(*ExtractionCylinder - SA_SDF *)
	ExtractionCylinder_DisabledSensorFault : BOOL;
	ExtractionCylinder_fault :BOOL;
	ExtractionCylinder_ActuatorFault : BOOL;

	(*AirCushion - SA_NF*)
	AirCushion_ActuatorFault : BOOL;



	(* PROCESSING *)

	(* RotaryTable - DANR_DF *)
	RotaryTable_EnabledSensorFault : BOOL;
	RotaryTable_DisabledSensorFault : BOOL;
	RotaryTable_fault :BOOL;
	RotaryTable_ActuatorFault : BOOL;

	(* ExpellingLever - SA_NF *)
	ExpellingLever_ActuatorFault : BOOL;



	(* INSPECTION UNIT *)

	(* CylinderToInspect - SA_NF *)
	CylinderToInspect_ActuatorFault : BOOL;



	(* DRILLING UNIT *)

	(* Drill_Machine - DA_DF *)
	Drill_Machine_EnabledSensorFault : BOOL;
	Drill_Machine_DisabledSensorFault : BOOL;
	Drill_Machine_fault :BOOL;
	Drill_Machine_ActuatorFault : BOOL;

	(* Drilling - SA_NF *)
	Drilling_ActuatorFault : BOOL;

	(* Holding - SA_NF *)
	Holding_ActuatorFault : BOOL;



	(* ASSEMBLY *)

	(*BlockingCylinder - SA_NF*)
	BlockingCylinder_ActuatorFault : BOOL;



	(* ITEM SUPPLY *)

	(*PistonSelector - DA-DF*)
	PistonSelector_EnabledSensorFault : BOOL;
	PistonSelector_DisabledSensorFault : BOOL;
	PistonSelector_fault :BOOL;
	PistonSelector_ActuatorFault : BOOL;

	(*ExtractCover - SA_DF*)
	ExtractCover_EnabledSensorFault : BOOL;
	ExtractCover_DisabledSensorFault : BOOL;
	ExtractCover_fault :BOOL;
	ExtractCover_ActuatorFault : BOOL;

	(*ExtractSpring - SA-DF*)
	ExtractSpring_EnabledSensorFault : BOOL;
	ExtractSpring_DisabledSensorFault : BOOL;
	ExtractSpring_fault :BOOL;
	ExtractSpring_ActuatorFault : BOOL;
END_VAR


(* Output from SignalControl *)
VAR_GLOBAL
	ResetSignalsEnable : BOOL;

	ImmediateStop_Signal : BOOL;
	OnPhaseStop_Signal : BOOL;
END_VAR
                                                                                               '           w   , , B ��        %   SignalManagement_LIB_Global_Variables ��d	��dw     ��������        �  VAR_GLOBAL CONSTANT
(* Signal type e relativa sua maschera *)
	SIGNAL_TYPE_MASK : WORD := 00007; (*ultimi 3 bit*)
	ALARM : WORD := 00001;
	ANOMALY : WORD := 00002;
	WARNING : WORD := 00003;
	INFORMATION : WORD := 00004;

(* Signal Reset Definitions *)
	SIGNAL_RESET_MASK : WORD := 15360; (*10 A 13 bit*)
	AUTO_RESET : WORD := 1024;
	AUTO_CONDITIONED_RESET : WORD := 08192;

(* To clean the variables *)
	NONE : DWORD := 16#0000;

 (*Dimensione massima dell'arrey contenente i segnali.
    Ovvero il numero massimo del "SignalCode" da associare ai segnali, generati dal file Python*)
	N :INT:=32;
END_VAR







(**** USEFULL TIPS FROM PALLI TO EXPANDING THE LYBRARY ***)
(*
VAR_GLOBAL CONSTANT
(* Condition *) (* Se usate vanno modificate in ward *)
	AUX_RESET : INT := 101;
	UNCONDITIONED_RESET : INT := 102;
	AUTO_RESET : INT := 103;
	AUTO_PRIORITY_RESET : INT := 104;
	AUTO_PROVISIONAL_RESET : INT := 105;
	AUTO_CONDITIONED_RESET : INT := 106;
END_VAR

VAR_GLOBAL
 	(* CONFIGURATION *)
	ConditionedResetSignals : BOOL;(* // TRUE*)
	TimePrioritySignals : BOOL; (* // TRUE*)
	SignalCodeDefault : WORD; (*// 0*)
	SignalCodeImpossible : WORD;(* // 11111*)
	SignalCodeError : WORD; (*// 9999*)
END_VAR
*)                                                                                               '           F   , ���           Simulation_Global_Variables ��d	��dF     ��������        ��  VAR_GLOBAL


(*------------------------------------------------------------------------------------------------------------------------------------------------*)
(*-----------------------------------VARIABILI DI VISUALIZZAZIONE DEGLI ELEMENTI----------------------------------------*)

(********************variabili utilizzate solo per la simulazione (NON per la logica di controllo)**********************)

(*---------Elementi presenti nel magazzino basi: il magazzino pu� contenere al pi� 8 elementi---------*)
	ElementOneCharged: BOOL; (*variabile che indica la presenza del primo elemento nel magazzino*)
	ElementTwoCharged: BOOL; (*secondo elemento caricato nel magazzino*)
	ElementThreeCharged: BOOL; (*terzo elemento caricato nel magazzino*)
	ElementFourCharged: BOOL; (*quarto elemento caricato nel magazzino*)
	ElementFiveCharged: BOOL; (*quinto elemento caricato nel magazzino*)
	ElementSixCharged: BOOL; (*sesto elemento caricato nel magazzino*)
	ElementSevenCharged: BOOL; (*settimo elemento caricato nel magazzino*)
	ElementEightCharged: BOOL; (*ottavo e ultimo elemento caricato nel magazzino*)

(*colori. I possibili colori degli elementi caricabili sono: Rosso, Nero e Argentato (o Metallico)*)
	Red: BOOL; (*� stato caricato un pezzo rosso*)
	Black: BOOL; (*� stato caricato un pezzo nero*)
	Silver: BOOL; (*� stato caricato un pezzo argentato*)

(*pezzi rossi*)
	ElementOneRed: BOOL; (*variabile che indica che il primo elemento caricato � rosso*)
	ElementTwoRed: BOOL; (*il secondo elemento caricato � rosso*)
	ElementThreeRed: BOOL; (*il terzo elemento caricato � rosso*)
	ElementFourRed: BOOL; (*il quarto elemento caricato � rosso*)
	ElementFiveRed: BOOL; (*il quinto elemento caricato � rosso*)
	ElementSixRed: BOOL; (*il sesto elemento caricato � rosso*)
	ElementSevenRed: BOOL; (*il settimo elemento caricato � rosso*)
	ElementEightRed: BOOL; (*l'ottavo elemento caricato � rosso*)

 (*pezzi metallici*)
	ElementOneSilver: BOOL; (*variabile che indica che il primo elemento caricato � argentato*)
	ElementTwoSilver: BOOL;
	ElementThreeSilver: BOOL;
	ElementFourSilver: BOOL;
	ElementFiveSilver: BOOL;
	ElementSixSilver: BOOL;
	ElementSevenSilver: BOOL;
	ElementEightSilver: BOOL;

 (*pezzi neri*)
	ElementOneBlack: BOOL; (*variabile che indica che il primo elemento caricato � nero*)
	ElementTwoBlack: BOOL;
	ElementThreeBlack: BOOL;
	ElementFourBlack: BOOL;
	ElementFiveBlack: BOOL;
	ElementSixBlack: BOOL;
	ElementSevenBlack: BOOL;
	ElementEightBlack: BOOL;

(*-----pezzi capovolti-----Pu� capitare che alcune basi vengano caricate capovolte nel magazzino: le variabili seguenti simulano questa ipotesi*)
	Redoverturned: BOOL; (*variabile che indica che l'elemento rosso � capovolto*)
	Silveroverturned: BOOL; (*variabile che indica che l'elemento argentato � capovolto*)
	Blackoverturned: BOOL; (*variabile che indica che l'elemento nero � capovolto*)

(*---------Elementi caricati capovolti nel magazzino basi--------*)
	ElementOneOverturned: BOOL; (*variabile che indica che il primo elemento del magazzino � stato caricato capovolto*)
	ElementTwoOverturned: BOOL; (*secondo elemento del magazzino caricato capovolto*)
	ElementThreeOverturned: BOOL; (*terzo elemento del magazzino caricato capovolto*)
	ElementFourOverturned: BOOL; (*quarto elemento del magazzino caricato capovolto*)
	ElementFiveOverturned: BOOL; (*quinto elemento del magazzino caricato capovolto*)
	ElementSixOverturned: BOOL; (*sesto elemento del magazzino caricato capovolto*)
	ElementSevenOverturned: BOOL; (*settimo elemento del magazzino caricato capovolto*)
	ElementEightOverturned: BOOL; (*ottavo elemento del magazzino caricato capovolto*)

(*----pezzi corti:
I pezzi chiari (rossi/met) sono alti, quelli neri, bassi. Pu� capitare che nel magazzino vengano caricati pezzi chiari ma bassi per difetti di costruzione:
le variabili seguenti simulano questa ipotesi*)

(*per la simulazione � stato creato solo il pezzo metallico corto: non � necessario creare anche quello rosso perch� il sistema non distingue tra rosso e metallico ma solo tra chiaro e nero*)
	Silvershort: BOOL; (*pezzo argentato corto*)

(*---------Elementi corti caricati nel magazzino basi--------*)
	ElementOneShort: BOOL; (*il primo elemento del magazzino � corto*)
	ElementTwoShort: BOOL;
	ElementThreeShort: BOOL;
	ElementFourShort: BOOL;
	ElementFiveShort: BOOL;
	ElementSixShort: BOOL;
	ElementSevenShort: BOOL;
	ElementEightShort: BOOL;

(*stringhe di segnalazione pezzo capovolto o corto: viene visualizzata una O se il pezzo � capovolto oppure una S se � corto. Niente altrimenti.*)
	ElementOneO: STRING; (*sul primo elemento del magazzino viene visualizzata stringa "O"(Overturned), oppure "S" (Short), oppure "" (pezzo normale)*)
	ElementTwoO: STRING;
	ElementThreeO: STRING;
	ElementFourO: STRING;
	ElementFiveO: STRING;
	ElementSixO: STRING;
	ElementSevenO: STRING;
	ElementEightO: STRING;

(*-----Elemento in Attesa: una volta espulso dal magazzino basi l'elemento rimane in attesa di essere caricato dal braccio rotante-----*)
	ElementWaitingCharged: BOOL; (*variabile che indica che � presente un elemento in attesa*)
	ElementWaitingRed: BOOL; (*variabile che indica che l'elemento in attesa � rosso*)
	ElementWaitingBlack: BOOL; (*variabile che indica che l'elemento in attesa � nero*)
	ElementWaitingSilver: BOOL; (*variabile che indica che l'elemento in attesa � argentato*)
	ElementWaitingOverturned: BOOL; (*variabile che indica che l'elemento in attesa � capovolto*)
	ElementWaitingShort: BOOL; (*variabile che indica che l'elemento in attesa � corto*)
	ElementWaitingO: STRING; (*stringa di segnalazione elemento in attesa capovolto o corto*)

(*------------Elemento sul Braccio rotante-------------*)
	ElementRotaryCharged: BOOL; (*variabile che indica che un elemento � caricato sul braccio rotante*)
	ElementRotaryRed: BOOL;
	ElementRotaryBlack: BOOL;
	ElementRotarySilver: BOOL;
	ElementRotaryOverturned: BOOL;
	ElementRotaryShort: BOOL;
	ElementRotaryO: STRING;

(*------------------Elemento in Verifica--------------------*)
	ElementVerificationCharged: BOOL;  (*variabile che indica che � presente un elemento nella stazione di verifica*)
	ElementVerificationRed: BOOL;
	ElementVerificationBlack: BOOL;
	ElementVerificationSilver: BOOL;
	ElementVerificationOverturned: BOOL;
	ElementVerificationShort: BOOL;
	ElementVerificationO: STRING;

(*------------------Elemento Misura----------------------*)
	ElementMeasureCharged: BOOL; (*variabile che indica che un elemento � giunto sotto al misuratore*)
	ElementMeasureRed: BOOL;
	ElementMeasureSilver: BOOL;
	ElementMeasureBlack: BOOL;
	ElementMeasureOverturned: BOOL;
	ElementMeasureShort: BOOL;
	ElementMeasureO: STRING;

(*--------------Elemento Cuscinetto d'aria--------------*)
	ElementAirCharged: BOOL;  (*variabile che indica la presenza di un elemento sulla guida a cuscinetto d'aria*)
	ElementAirRed: BOOL;
	ElementAirSilver: BOOL;
	ElementAirBlack: BOOL;
	ElementAirOverturned: BOOL;
	ElementAirO: STRING;



																			(*ATTUATORI E SENSORI*)
(*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*)

(* -------------MODULO MAGAZZINO DI STOCCAGGIO (magazzino basi) -----------*)

(*variabili associate ad attuatori e sensori: (***utilizzate nella logica di controllo***)*)
	EmptyWarehouse:BOOL;  (*sensore magazzino pezzi vuoto*)
	CylinderExtractionLoadInExtensivePosition:BOOL; (*sensore cilindro di estrazione in posizione estratta*)
	CylinderExtractionLoadInRetroactivePosition:BOOL; (*sensore cilindro di estrazione in posizione retratta*)
	CylinderExtractsLoadFromWarehouse:BOOL; (*comando di estrazione pezzo dal magazzino *)

(*variabili della simulazione *)
	CylinderPosition: INT; (* variabile utilizzata per simulare il movimento del cilindro di estrazione *)
	ElementPosition: INT; (* variabile utilizzata per simulare il movimento dell'elemento in uscita dal magazzino *)
	CylinderBehind:BOOL; (*variabile che indica che il cilindro sta tornando indietro *)

(*----------------- MODULO DI TRASFERIMENTO (braccio rotante) ---------------*)

(*variabili associate ad attuatori e sensori: (***utilizzate nella logica di controllo***)*)
	RotaryMakerVsWarehouse:BOOL;  (*comando rotary maker verso magazzino pezzi *)
	RotaryMakerVsVerification:BOOL;  (*comando rotary maker verso stazione di verifica *)
	RotaryMakerInPositionWarehouse:BOOL;  (*sensore di finecorsa rotary maker in posizione magazzino pezzi *)
	RotaryMakerInPositionVerification:BOOL;  (*sensore di finecorsa rotary maker in stazione di verifica *)
	VacuumGenerator:BOOL;  (*comando generatore di vuoto *)
	VacuumGeneratorOk:BOOL;  (*sensore di vuoto*)
	ExpulsionAirVacuum:BOOL;  (*comando getto d'aria per espulsione *)

(*variabili della simulazione *)
	RotaryPosition: INT; (* variabile utilizzata per simulare il movimento del rotary maker *)
       ReadyForRotaryMaker: BOOL; (* variabile di simulazione che indica la disponibilit� di un pezzo per il braccio rotante*)
	VacuumGeneratorSim: BOOL; (*simulazione generatore di vuoto*)
	ExpulsionAirVacuumVis:BOOL; (*visualizzazione getto d'aria per espulsione *)

(*simulazione valore iniziale braccio rotante: *)
	RotaryPositionInitialVis: INT; (*variabile intera che indica la posizione in cui si trova il braccio rotante prima della simulazione*)
	FLAGRotaryPosition: BOOL := TRUE;

(* -------------------------------------STAZIONE DI VERIFICA------------------------------- *)

(*variabili associate ad attuatori e sensori: (***utilizzate nella logica di controllo***)*)
	VerificationBusy: BOOL; (*sensore a fotocellula per interferenza braccio rotante-ascensore*)
	ToLiftCylinderToMeasureLoad:BOOL; (*comando modulo di sollevamento verso l'alto *)
	ToLowerCylinderToMeasureLoad:BOOL; (*comando modulo di sollevamento verso il basso *)
	CylinderUpToMeasureLoad:BOOL;  (*sensore finecorsa modulo di sollevamento in alto *)
	CylinderDownToMeasureLoad:BOOL; (*sensore finecorsa modulo di sollevamento in basso *)
	ToExtendCylinderOfExtractionVsGuide:BOOL; (*comando espulsione pezzo *)
	CylinderOfExtractionInRetroactivePosition:BOOL; (*sensore finecorsa cilindro di estrazione in posizione retratta *)
       CylinderOfExtractionInExtensivePosition:BOOL; (*sensore finecorsa cilindro di estrazione in posizione estesa (Virtuale) *)
       ResetVirtualSensorElevator: BOOL; (*Serve a resettare il sensore virtuale dell'elevatore a seguito di un plausibile Fault*)

       AirCushion:BOOL; (*comando cuscinetto d'aria*)
	ReadyLoadForVerification:BOOL; (*sensore presenza pezzo alla base della stazione di verifica *)
	ColourMeasurement:BOOL; (*sensore di rilevazione colore: 0 nero, 1 rosso/metallico *)
	MeasurementNotOk: BOOL; (*sensore di misuarazione altezza. Uscita del misuratore: 1 pezzo alto, 0 pezzo basso*)

(*variabili della simulazione *)
	LiftPosition: INT; (* variabile utilizzata per simulare il movimento del modulo di sollevamento *)
	CylinderOfExtractionPosition:INT; (* variabile utilizzata per simulare il movimento del cilindro espulsione *)
	MisuratorPosition: INT; (* variabile utilizzata per simulare il movimento del misuratore*)
	MovementElementAir: INT; (* variabile utilizzata per simulare il movimento dell'elemento sul cuscino d'aria*)
	AirCushionVis: BOOL; (*visualizzazione cuscinetto d'aria*)
	ElementAirVsRotaryTable: BOOL; (*variabile di simulazione che notifica l'arrivo dell'elemento d'aria sulla tavola rotante*)

(*simulazione valore iniziale modulo di sollevamento*)
	LiftPositionInitialVis: INT;  (*variabile intera che indica la posizione in cui si trova l'ascensore prima della simulazione*)
	FLAGLiftPosition: BOOL:=TRUE;

(*STAZIONE DI LAVORAZIONE*)

	(* MODULO TAVOLA ROTANTE*)
	RotaryTableMotor:BOOL; (* COMANDO DI ATTUAZIONE: tavola rotante *)
	AlignementRotaryTableWithPositionings:BOOL; (* SENSORE: tavola allineata con le postazioni *)
       AvailableLoadForWorkingStation:BOOL; (* SENSORE DI PRESENZA: load presente nella prima stazione della giostra*)
       AvailableLoadInControlPositioning:BOOL; (*SENSORE DI PRESENZA: load presente in postazione di controllo*)
       AvailableLoadInDrillingPositioning:BOOL; (*SENSORE DI PRESENZA: load presente nel modulo di foratura*)

	(* MODULO DI CONTROLLO*)
	ToLowerCylinderToInspectLoad:BOOL; (* COMANDO DI ATTUAZIONE: L'unit� di controllo viene abbassata *)
	InControlLoadInWrongPositionToBeDrilled: BOOL; (*SENSORE per il rilevamento del corretto orientamento della base*)

      	(* MODULO DI FORATURA.*)
	ToLiftDrillingUnit:BOOL; (*COMANDO DI ATTUAZIONE:  l'unit� di foratura viene abbassata *)
	ToLowerDrillingUnit:BOOL; (*COMANDO DI ATTUAZIONE: l'unit� di foratura viene sollevata *)
	DrillingUnitUp:BOOL; (* SENSORE: unit� di foratura in posizione sollevata*)
	DrillingUnitDown:BOOL; (* SENSORE: unit� di foratura sulla base del pezzo *)
	DrillingUnitActive:BOOL; (* COMANDO DI ATTUAZIONE: unit� di foratura ruota in senso orario *)
	BlockingCylinderForwardInDrillingPositioning:BOOL; (*COMANDO DI ATTUAZIONE: pistone di bloccaggio pezzo*)
       DrillingUnitClockWise:BOOL;
       (* MODULO DI ESPULSIONE.*)
       ExpellingLeverActive:BOOL; (*COMANDO DI ATTUAZIONE:  leva espelle pezzo *)


	(*STAZIONE DI ASSEMBLAGGIO*)

 	(* MODULO MOLLE.*)
	ToExtractSpringInAssemblyStation:BOOL; (* COMANDO DI ATTUAZIONE del cilindro che preleva la molla dal magazzino *)
	ToExtractSpringInAssemblyStationInExtensivePosition:BOOL; (* SENSORE di fine corsa che indica che il cilindro � in posizione estesa(di riposo)*)
	ToExtractSpringInAssemblyStationInRetroactivePosition:BOOL; (*SENSORE di fine corsa che indica che il cilindro � in posizione ritratta*)

	(* MODULO MAGAZZINO  PISTONI.*)
	PistonSelectorGoOnTheRight:BOOL; (*COMANDO DI ATTUAZIONE: magazzino pistoni ruota verso destra (preleva pistone NERO)*)
	PistonSelectorGoOnTheLeft:BOOL; (*COMANDO DI ATTUAZIONE: magazzino pistoni ruota verso sinistra(preleva pistone GRIGIO)*)
	PistonSelectorIsOnTheRight:BOOL; (* SENSORE: magazzino pistoni ruotato completamente a destra *)
	PistonSelectorIsOnTheLeft:BOOL; (* SENSORE: magazzino pistoni ruotato completamente a sinistra *)

	(* MODULO STOCCAGGIO COPERCHI.*)
	ToExtractCoverInAssemblyStationForward:BOOL; (*COMANDO DI ATTUAZIONE: cilindro estrae cover *)
	EmptyCoverHouseInAssemblyStation: BOOL; (*SENSORE: segnala se sono presenti dei coperchi nel magazzino*)
	ToExtractCoverInAssemblyStationInRetroactivePosition:BOOL; (*SENSORE:  cilindro di estrazione in posizione retratta *)
	ToExtractCoverInAssemblyStationInExtensivePosition:BOOL; (*SENSORE: cilindro di estrazione in posizione estesa *)

	(*STAZIONE1 DOPO LA GIOSTRA*)
	AvailableLoadForRobot:BOOL; (*SENSORE: nella stazione1 dopo la giostra, indica la presenza di una base pronta per l'assemblaggio*)
	(*STAZIONE DI ASSEMBLAGGIO PEZZO*)
	BlockingCylinderForwardInAssemblyStation:BOOL;  (*COMANDO DI ATTUAZIONE: cilindro di boccaggio, SE TRUE SI SBLOCCA il pezzo *)
	(*POSIZIONE INIZIALE ROBOT*)
	RobotGoToInitialPosition: BOOL; (*COMANDO DI ATTUAZIONE: porta il robot nella posizione iniziale*)
	RobotInInitialPosition: BOOL; (*SENSORE: indica che il robot � arrivato nella posizione iniziale*)

	(*STAZIONE DI ASSEMBLAGGIO*)
	RobotTakeBlackLoad: BOOL; (*COMANDO DI ATTUAZIONE : impone al robot la sequenza di istruzioni tale da prelevare una base Nera*)
	RobotTakeRedSilverLoad: BOOL; (*COMANDO DI ATTUAZIONE : impone al robot la sequenza di istruzioni tale da prelevare una base Rossa-Argento*)
	RobotTakeLoadToDiascard: BOOL; (*COMANDO DI ATTUAZIONE : impone al robot la sequenza di istruzioni tale da prelevare una base Capovolta*)
	RobotTakeCurrentLoadToAssembly: BOOL; (*COMANDO DI ATTUAZIONE : impone al robot la sequenza di istruzioni tale da portare il robot nella stazione di assemblaggio*)

	RobotInAssemblyUnit: BOOL; (*SENSORE: robot in posizione di assemblaggio*)
	(*STAZIONE DEI PISTONI*)
	RobotGoToPistonHouse: BOOL;  (*COMANDO DI ATTUAZIONE: porta il robot nella stazione dei pistoni*)
	RobotInPistonWarehouse: BOOL;  (*SENSORE: robot nella stazione dei pistoni*)
	(*STAZIONE DELLE MOLLE*)
	RobotGoToSpringHouse: BOOL; (*COMANDO DI ATTUAZIONE: porta il robot nella stazione delle molle*)
	RobotInSpringWarehouse:BOOL:=FALSE; (* SENSORE: robot nella stazione delle molle *)
	(*STAZIONE DEI COPERCHI*)
	RobotGoToCoverHouse: BOOL; (*COMANDO DI ATTUAZIONE: porta il robot nella stazione dei coperchi*)
	RobotInCoverWarehouse:BOOL;  (*SENSORE: robot nella stazione dei coperchi*)

	(*VARIABILI DI CONTROLLO utilizzate anche nella simulazione*)
	Fault: BOOL; (*Variabile di visualizzazione: se true rende visibile il display "Fault" *)
	FaultDetected: BOOL; (*GUASTO RILEVATO, per far ripartire il sistema dopo un fault che non necessita di riavvio*)
	EmergencyRobot: BOOL; (*Variabile per fermare il robot in caso di emergenza*)
(*xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx*)


(*------------------------------------------------------------------------------------------------------------------------------------------------------------------------*)
(*-----------------------------------------------------------------PULSANTIERE----------------------------------------------------------------------------------*)

(**************variabili utilizzate nella logica di controllo (e nella simulazione)***************)

(* PULSANTIERA DI COMANDO*)

	Start:BOOL; (*Pulsante di start*)
	LightStart:BOOL; (*Luce del pulsante di start*)
	Reset:BOOL; (*Pulsante di reset*)
	LightReset:BOOL; (*Luce del pulsante di reset*)
	Stop:BOOL:=TRUE; (*Pulsante di stop*)
	FreezeStop:BOOL:=TRUE; (*Pulsate di "congelamento" del sistema*)
	EmergencyStop:BOOL:=TRUE; (*Pulsante di stop di emergenza*)
	FreezeStopPuls:BOOL:=TRUE; (*Pulsate di "congelamento" del sistema*)

(* PULSANTIERA DI CONTROLLO*)

	LightEmptyWarehouse: BOOL; (*Luce magazzino basi vuoto*)
	FullWarehouse:BOOL; (*Pulsante magazzino basi riempito*)
	LightUpsideDownLoadInExpelling:BOOL; (*Luce pezzo capovolto*)
	LightEmptyCoverHouse:BOOL; (*Luce magazzino coperchi vuoto*)
	FullWarehouseInAssemblyStation:BOOL; (*Pulsante magazzino coperchi riempito*)
	ToWorkBlackLoad:BOOL; (*Selettore colore pezzi da lavorare*)
	ToWorkBlackOrRedLoad:BOOL; (*Selettore tipo pezzi da lavorare*)
	UpsideDownLoadRemovedInExpelling:BOOL; (*Pulsante pezzo capovolto rimosso*)
	LightRobotInMovement:BOOL; (*Luce robot in movimento*)
	LightRobotServoON:BOOL; (*Luce motori robot accesi*)
	LightRedLoad:BOOL; (*Luce pezzo rosso/metallico*)
	LightBlackLoad:BOOL; (*Luce pezzo nero*)

(************************************************************************************************)

(* VARIABILI ASSOCIATE ALLE PULSANTIERE ---------solo per la simulazione--------*)

(*pulsantiera virtuale*)
	ToWorkBlackLoadPuls: BOOL;
	ToWorkBlackOrRedLoadPuls: BOOL;
	FullWarehousePuls: BOOL;
	UpsideDownLoadRemovedInExpellingPuls:BOOL;
	FullWarehouseInAssemblyStationPuls: BOOL;
	EnableVirtualBox: BOOL; (*pulsante di abilitazione pulsantiera virtuale (disabilitazione pulsantiera fisica)*)
	Remove: BOOL; (*pulsante di rimozione di tutti i pezzi dall'impianto*)

(*pulsantiera fisica*)
	ToWorkBlackLoadPin: BOOL;
	ToWorkBlackOrRedLoadPin: BOOL;
	FullWarehousePin: BOOL;
	UpsideDownLoadRemovedInExpellingPin:BOOL;
	FullWarehouseInAssemblyStationPin: BOOL;

(*-----------------------------------------------------------------------------------------------------------------------------------------------------------*)
(*---------------------------------------------------------------------GUASTI---------------------------------------------------------------------------*)

(* --------------------SIMULAZIONE guasti--------------------- *)

(****variabili utilizzate solo per la simulazione****)

(*MODULO MAGAZZINO DI STOCCAGGIO (magazzion basi)*)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	EmptyWarehouseBlockHigh: BOOL;
	EmptyWarehouseBlockLow: BOOL;
	CylinderExtractionLoadInRetroactivePositionBlockLow: BOOL;
	CylinderExtractionLoadInRetroactivePositionBlockHigh: BOOL;
	CylinderExtractionLoadInExtensivePositionBlockLow: BOOL;
	CylinderExtractionLoadInExtensivePositionBlockHigh: BOOL;
(*attuatori: possono essere bloccati o bloccato alti (a 1)*)
	CylinderExtractsLoadFromWarehouseBlock: BOOL;
	CylinderExtractsLoadFromWarehouseBlockHigh: BOOL;

(* MODULO DI TRASFERIMENTO (braccio rotante) *)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	VacuumGeneratorOkBlockHigh: BOOL;
	VacuumGeneratorOkBlockLow: BOOL;
	RotaryMakerInPositionWarehouseBlockHigh: BOOL;
	RotaryMakerInPositionWarehouseBlockLow: BOOL;
	RotaryMakerInPositionVerificationBlockHigh: BOOL;
	RotaryMakerInPositionVerificationBlockLow: BOOL;

(*attuatori: possono essere bloccati o bloccati alti (a 1)*)
	VacuumGeneratorBlock: BOOL;
	VacuumGeneratorBlockHigh: BOOL;
	ExpulsionAirVacuumBlock: BOOL;
	ExpulsionAirVacuumBlockHigh: BOOL;
	RotaryMakerVsWarehouseBlock: BOOL;
	RotaryMakerVsWarehouseBlockHigh: BOOL;
	RotaryMakerVsVerificationBlock: BOOL;
	RotaryMakerVsVerificationBlockHigh: BOOL;

(* STAZIONE DI VERIFICA *)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	VerificationBusyBlockLow: BOOL;
	VerificationBusyBlockHigh: BOOL;
	ReadyLoadForVerificationBlockHigh: BOOL;
	ReadyLoadForVerificationBlockLow: BOOL;
	CylinderDownToMeasureLoadBlockHigh: BOOL;
	CylinderDownToMeasureLoadBlockLow: BOOL;
	CylinderUpToMeasureLoadBlockHigh: BOOL;
	CylinderUpToMeasureLoadBlockLow: BOOL;
	CylinderOfExtractionInRetroactivePositionBlockHigh: BOOL;
	CylinderOfExtractionInRetroactivePositionBlockLow: BOOL;
	ColourMeasurementBlockHigh: BOOL;
	ColourMeasurementBlockLow: BOOL;
	MeasurementNotOkBlockHigh: BOOL;
	MeasurementNotOkBlockLow: BOOL;

(*attuatori: possono essere bloccati o bloccati alti (a 1)*)
	ToLiftCylinderToMeasureLoadBlock: BOOL;
	ToLiftCylinderToMeasureLoadBlockHigh: BOOL;
	ToLowerCylinderToMeasureLoadBlock: BOOL;
	ToLowerCylinderToMeasureLoadBlockHigh: BOOL;
	ToExtendCylinderOfExtractionVsGuideBlock: BOOL;
	ToExtendCylinderOfExtractionVsGuideBlockHigh: BOOL;
	AirCushionBlock: BOOL;
	AirCushionBlockHigh: BOOL;


(*-------------------------RILEVAZIONE guasti ---------------------------*)

(************variabili utilizzate nella logica di controllo (e nella simulazione)***********)

(* MODULO MAGAZZINO DI STOCCAGGIO (magazzion basi) *)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	EmptyWarehouseBlockedHigh: BOOL;
	EmptyWarehouseBlockedLow: BOOL;
	CylinderExtractionLoadInRetroactivePositionBlockedLow: BOOL;
	CylinderExtractionLoadInRetroactivePositionBlockedHigh: BOOL;
	CylinderExtractionLoadInExtensivePositionBlockedLow: BOOL;
	CylinderExtractionLoadInExtensivePositionBlockedHigh: BOOL;

(*attuatori: possono essere bloccati o bloccati alti (a 1)*)
	CylinderExtractsLoadFromWarehouseBlocked: BOOL;
	CylinderExtractsLoadFromWarehouseBlockedHigh: BOOL;

(* MODULO DI TRASFERIMENTO (braccio rotante) *)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	VacuumGeneratorOkBlockedHigh: BOOL;
	VacuumGeneratorOkBlockedLow: BOOL;
	RotaryMakerInPositionWarehouseBlockedHigh: BOOL;
	RotaryMakerInPositionWarehouseBlockedLow: BOOL;
	RotaryMakerInPositionVerificationBlockedHigh: BOOL;
	RotaryMakerInPositionVerificationBlockedLow: BOOL;

(*attuatori: possono essere bloccati o bloccati alti (a 1)*)
	VacuumGeneratorBlocked: BOOL;
	VacuumGeneratorBlockedHigh:BOOL;
	ExpulsionAirVacuumBlocked: BOOL;
	ExpulsionAirVacuumBlockedHigh:BOOL;
	RotaryMakerVsWarehouseBlocked: BOOL;
	RotaryMakerVsWarehouseBlockedHigh: BOOL;
	RotaryMakerVsVerificationBlocked: BOOL;
	RotaryMakerVsVerificationBlockedHigh: BOOL;

(* STAZIONE DI VERIFICA *)

(*sensori: possono essere bloccati alti (a 1) o bloccati bassi (a 0)*)
	VerificationBusyBlockedLow: BOOL;
	VerificationBusyBlockedHigh: BOOL;
	ReadyLoadForVerificationBlockedHigh: BOOL;
	ReadyLoadForVerificationBlockedLow: BOOL;
	CylinderDownToMeasureLoadBlockedHigh: BOOL;
	CylinderDownToMeasureLoadBlockedLow: BOOL;
	CylinderUpToMeasureLoadBlockedHigh: BOOL;
	CylinderUpToMeasureLoadBlockedLow: BOOL;
	CylinderOfExtractionInRetroactivePositionBlockedHigh: BOOL;
	CylinderOfExtractionInRetroactivePositionBlockedLow: BOOL;
	ColourMeasurementBlockedHigh: BOOL;
	ColourMeasurementBlockedLow: BOOL;
	MeasurementNotOkBlockedHigh: BOOL;
	MeasurementNotOkBlockedLow: BOOL;

(*attuatori: possono essere bloccati o bloccati alti (a 1)*)
	ToLiftCylinderToMeasureLoadBlocked: BOOL;
	ToLiftCylinderToMeasureLoadBlockedHigh:BOOL;
	ToLowerCylinderToMeasureLoadBlocked: BOOL;
	ToLowerCylinderToMeasureLoadBlockedHigh: BOOL;
	ToExtendCylinderOfExtractionVsGuideBlocked: BOOL;
	ToExtendCylinderOfExtractionVsGuideBlockedHigh:BOOL;
	AirCushionBlocked: BOOL;
	AirCushionBlockedHigh:BOOL;

(*---------------------------------------------------------------------------------------------------------------------------------*)
(* ALTRE VARIABILI IMPIEGATE NELLA LOGICA DI CONTROLLO (e utilizzate anche nella simulazione) *)

	UncorrectComparison: BOOL; (*comparazione tra l'altezza del pezzo e il colore rilevato: 1 se la misura � incompatibile con il colore, 0 altrimenti*)
	VirtualCylinderDownToMeasureLoad: BOOL; (*sensore virtuale finecorsa inferiore ascensore. Quando l'ascensore scende, va a 1 sul secondo fronte di discesa del sensore CylinderDownToMeasureLoad*)




	(*variabili di controllo utilizzate dalla mia politica di controllo*)
	Stopped: BOOL;
	EndInit: BOOL; (*fine inizializzazione*)
	EndAssembling: BOOL;
	EndProcessing: BOOL;
	EmptyCover: BOOL;

																	(*VARIABILI DI SIMULAZIONE*)
(*___________________________________________________________________________________________________________________________________________*)



	(*VARIABILI DI INIZIALIZZAZIONE CASUALE DELLE GRANDEZZE DEGLI ATTUATORI NON PNEUMATICI*)
	RotaryVisIn: INT;
	DrillingUnitHeight: INT;



	(*STAZIONE DI LAVORAZIONE*)

	(* MODULO TAVOLA ROTANTE*)
	RotaryTablePosition:INT; (*VARIABILE DI VISUALIZZAZIONE:  incremento della posizione della tavola(resettata al primo allineamento) *)
	RotaryTableVisual:INT; (*VARIABILE DI VISUALIZZAZIONE:  incremento della posizione della tavola(con offset iniziale)*)
	Alligneed: BOOL; (*VARIABILE DI SIMULAZIONE:  indica l'allineamento della tavola*)

	(*VARIABILI PER LA GESTIONE DEI COLORI DEI PEZZI NELLE STAZIONI DELLA GIOSTRA*)

	(*                                                                              NERO            BIANCO              ROSSO                GRIGIO     *)
	ColorCircle : ARRAY[1..4] OF DWORD := 16#00000000,16#00FFFFFF, 16#000000FF, 16#00E1E1E1;
	Color1: DWORD:=16#00FFFFFF;
	Color2: DWORD:=16#00FFFFFF;
	Color3: DWORD:=16#00FFFFFF;
	Color4: DWORD:=16#00FFFFFF;
	Color5: DWORD:=16#00FFFFFF;
	Color6: DWORD:=16#00FFFFFF;

	ElementOneTableCharged:BOOL;
	ElementOneTableRed:BOOL;
	ElementOneTableBlack:BOOL;
	ElementOneTableSilver:BOOL;
	ElementOneTableOverturned:BOOL;
	LinearTablePosition1: LREAL;

	ElementTwoTableCharged: BOOL;
	ElementTwoTableRed:BOOL;
	ElementTwoTableBlack:BOOL;
	ElementTwoTableSilver:BOOL;
	ElementTwoTableOverturned:BOOL;
	LinearTablePosition2:LREAL;

	ElementThreeTableCharged:BOOL;
	ElementThreeTableRed:BOOL;
	ElementThreeTableBlack:BOOL;
	ElementThreeTableSilver:BOOL;
	ElementThreeTableOverturned:BOOL;
	LinearTablePosition3: LREAL;

	ElementFourTableCharged: BOOL;
	ElementFourTableRed:BOOL;
	ElementFourTableBlack:BOOL;
	ElementFourTableSilver:BOOL;
	ElementFourTableOverturned:BOOL;
	LinearTablePosition4:LREAL;

	ElementFiveTableCharged:BOOL;
	ElementFiveTableRed:BOOL;
	ElementFiveTableBlack:BOOL;
	ElementFiveTableSilver:BOOL;
	ElementFiveTableOverturned:BOOL;
	LinearTablePosition5: LREAL;

	ElementSixTableCharged: BOOL;
	ElementSixTableRed:BOOL;
	ElementSixTableBlack:BOOL;
	ElementSixTableSilver:BOOL;
	ElementSixTableOverturned:BOOL;
	LinearTablePosition6:LREAL;

	(*VARIABILI PER VISUALIZZARE 'O' SE IL PEZZO E' OVERTURNED*)
	ElementOneTableO: STRING;
	ElementTwoTableO: STRING;
	ElementThreeTableO: STRING;
	ElementFourTableO: STRING;
	ElementFiveTableO: STRING;
	ElementSixTableO: STRING;
	ElementSleighO: STRING;
	ElementStation1RobotO: STRING;

	(* MODULO DI CONTROLLO*)
	InspectPosition:INT; (*VARIABILE DI VISUALIZZAZIONE: incremento posizione utensile per il controllo*)
	InspectDepth: INT; (*variabile utilizzata per simulare le 2 differenti profondit� a cui si porta l'utensile di controllo in caso di base corretta o di base Overturned*)
	FLAGInControlLoadInWrongPositionToBeDrilled: BOOL; (*variabile utilizzata per mantenere acceso il led del sensore "InControlLoadInWrongPositionToBeDrilled" in caso di bloccaggio alto*)
      	(* MODULO DI FORATURA.*)
	DrillingUnitPosition:INT; (*VARIABILE DI VISUALIZZAZIONE: incremento della posizione dell'utensile di foratura*)
	BlockingActuator:INT;(*VARIABILE DI VISUALIZZAZIONE: incremento pistone di bloccaggio*)
	DrillingToolColor: BOOL; (*VARIABILE DI VISUALIZZAZIONE : cambio colore alla punta del trapano quando quest'ultimo viene attivato*)

       (* MODULO DI ESPULSIONE.*)
	AvailableLoadInExpulsionPositioning: BOOL:=FALSE; (*variabile fittizia per segnalare l'arrivo della base nella 4stazione della giostra*)
	LeverPosition:INT:=0; (*VARIABILE DI VISUALIZZAZIONE: incremento rotazione leva*)

	(*Slitta tra lavorazione e assemblaggio*)
	ElementSleighCharged: BOOL;
	ElementSleighRed: BOOL;
	ElementSleighSilver: BOOL;
	ElementSleighBlack: BOOL;
	ElementSleighOverturned:BOOL;
	MovementElementSleigh: INT;


	(*STAZIONE DI ASSEMBLAGGIO*)

	(*Stazione1 della stazione di assemblaggio*)
	ElementStation1RobotCharged: BOOL;
	ElementStation1RobotRed: BOOL;
	ElementStation1RobotBlack: BOOL;
	ElementStation1RobotSilver: BOOL;
	ElementStation1RobotOverturned: BOOL;

	(*STAZIONE DI CONTENIMENTO DEI PEZZI SCARICHI*)
	CanDiscard:BOOL:=FALSE;
	(*visualizzazione della base che finisce nel cesto degli scarti*)
	ElementToDiscard: BOOL;
	ElementToDiscardPosition: INT:=0;
	ElementToDiscardRed: BOOL;
	ElementToDiscardSilver: BOOL;
	ElementToDiscardBlack: BOOL;

	(*STAZIONE DI ASSEMBLAGGIO PEZZO*)
	ElementAssemblyCharged: BOOL;
	ElementAssemblyRed: BOOL;
	ElementAssemblySilver: BOOL;
	ElementAssemblyBlack: BOOL;
	CoverAssembly: BOOL;
	PistonBlackAssembly: BOOL;
	PistonSilverAssembly: BOOL;
	SpringAssembly: BOOL;

 	(* MODULO MOLLE.*)
	VisualSpring:BOOL; (* VARIABILE DI VISUALIZZAZIONE: moto traslatorio molla*)
	Spring1:BOOL;  (* VARIABILE DI VISUALIZZAZIONE: *)
      	Spring2:BOOL;  (* VARIABILE DI VISUALIZZAZIONE: *)
	Spring3:BOOL;  (* VARIABILE DI VISUALIZZAZIONE: *)
	Spring4:BOOL;  (* VARIABILE DI VISUALIZZAZIONE: *)
	Spring5:BOOL;  (*VARIABILE DI VISUALIZZAZIONE: *)
	Spring6:BOOL;  (*VARIABILE DI VISUALIZZAZIONE: *)
	Spring7:BOOL;  (* VARIABILE DI VISUALIZZAZIONE: *)
	Spring8:BOOL; (* VARIABILE DI VISUALIZZAZIONE: *)
	ExtractSpringPosition: INT:=40; (*la molla mantiene il cilindro in posizione di riposo NA*)
	EmptySpringWarehouse: BOOL:=TRUE;
	SpringNumber: INT;
	SpringLoad: BOOL:=FALSE;


	(* MODULO MAGAZZINO  PISTONI.*)
	PistonBlack8: BOOL;
	PistonBlack7: BOOL;
	PistonBlack6: BOOL;
	PistonBlack5: BOOL;
	PistonBlack4: BOOL;
	PistonBlack3: BOOL;
	PistonBlack2: BOOL;
	PistonBlack1: BOOL;
	VisualPistonBlack: BOOL; (*VARIABILE DI VISUALIZZAZIONE: traslazione del pistoncino*)
	PistonBlackNumber:INT;

	PistonSilver1: BOOL;
	PistonSilver2: BOOL;
	PistonSilver3: BOOL;
	PistonSilver4: BOOL;
	PistonSilver5: BOOL;
	PistonSilver6: BOOL;
	PistonSilver7: BOOL;
	PistonSilver8: BOOL;
	VisualPistonSilver: BOOL;
	PistonSilverNumber:INT;

	PistonSelectorPosition: INT:=20;(*VARIABILE DI VISUALIZZAZIONE:  movimento orizzontale del cilindro di estrazione pistoni*)
	VisualPistonBlackPosition: INT;(*VARIABILE DI VISUALIZZAZIONE:  movimento orizzontale del pistone nero*)
	VisualPistonSilverPosition: INT;(*VARIABILE DI VISUALIZZAZIONE:  movimento orizzontale del pistone grigio*)
	PistonLoad: BOOL; (*carica contemporaneamente i pistoni neri e grigi*)
	PistonBlackLoad: BOOL; (*carica pistoni black*)
	PistonSilverLoad: BOOL; (*carica pistoni silver*)
	(* magazzino pistoni nero o rosso vuoto. IN REALTA' NON C'E' ALCUN SENSORE DI PRESENZA*)
	EmptyBlackOrRedPistonWarehouseInAssemblyStation:BOOL:=TRUE;
	EmptyPistonBlackWarehouse: BOOL:=TRUE;
	EmptyPistonSilverWarehouse: BOOL:=TRUE;

	(* MODULO STOCCAGGIO COPERCHI.*)
	VisualCover:BOOL;
	Cover1:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover2:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover3:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover4:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover5:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover6:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover7:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	Cover8:BOOL; (* variabile utilizzata per rendere visibile un coperchio nel magazzino cover *)
	ExtractCoverPosition: INT:=0;
	CoverLoad: BOOL;
	CoverNumber: INT;

	(* STAZIONE DI ASSEMBLAGGIO PEZZO - MODULO BLOCCAGGIO*)
	BlockingCylinderActiveInAssemblyStation:BOOL; (* cilindro di bloccaggio blocca pezzo *)
	CylinderPositionInAssemblyUnit: INT;

	(* ROBOT.*)
	StopProgramRunning: BOOL; (*VARIABILE DI VISUALIZZAZIONE: associata al pulsante omonimo, simula il bloccaggio del programma CosiRop*)
	RobotGoVerticalPosition: INT:=0; (*VARIABILE DI VISUALIZZAZIONE: incremento di posizione lungo l'asse verticale*)
	RobotGoHorizontalPosition: INT:=0; (*VARIABILE DI VISUALIZZAZIONE: incremento di posizione lungo l'asse orizzontale*)
	RobotProgramRunning: BOOL;
	(*5 bit di output dal PLC al robot, le configurazioni di bit relative alle istruzioni sono implementate nel program(OutputBitConfiguration)*)
	Bit0Output: BOOL;
	Bit1Output: BOOL;
	Bit2Output: BOOL;
	Bit3Output: BOOL;
	Bit4Output: BOOL;
	(*4 bit di input dal robot al PLC, le configurazioni di bit relative ai sensori virtuali sono implementate nel program(OutputBitConfiguration)*)
	Bit0Input: BOOL;
	Bit1Input: BOOL;
	Bit2Input: BOOL;
	Bit3Input: BOOL;

	(*VARIABILI DI VISUALIZZAZIONE: per rendere visibili gli elementi nell'end-effector*)
	EndEffectorPosition: INT:=0; (*VARIABILE DI VISUALIZZAZIONE: incremento di posizione "slide" del end effector*)
	ElementInEndEffectorO: STRING; (*aggiunto, O nella base nell'end-effector*)
	ElementInEndEffector: BOOL; (*aggiunto, rende visibile la base nell'end-effector*)
	ElementInEndEffectorBlack: BOOL;
	ElementInEndEffectorRed: BOOL;
	ElementInEndEffectorSilver: BOOL;
	ElementInEndEffectorOverturned: BOOL;
	PistonBlackInEndEffector: BOOL;
	PistonSilverInEndEffector: BOOL;
	SpringInEndEffector: BOOL;
	CoverInEndEffector: BOOL;

	(*VARIABILI UTILIZZATE PER LO SCAMBIO DI PEZZI TRA END-EFFECTOR E MAGAZZINI*)
	SpringExtract: BOOL;
	PistonSilverExtract: BOOL;
	PistonBlackExtract:BOOL;
	CoverExtract: BOOL;

	(*VARIABILE CHE VIENE MESSA A TRUE QUANDO UN ELEMENTO E' COMPLETAMENTE ASSEMBLATO*)
	ElementAssembled: BOOL;

	(*Variabili per visualizzazione cestino scarti-pezzi finiti*)
	CanColour: BOOL; (*variabile utilizzata per cambiare il colore del cestino a seconda che esso venga utilizzato per gli scarti o per i pezzi finiti*)
	CanText: STRING; (*variabile utilizzata per cambiare il testo nel cestino a seconda che esso venga utilizzato per gli scarti o per i pezzi finiti*)
	CoverToDiscard: BOOL; (*variabile utilizzata per visualizzare il coperchio nella base che finisce nel cestino*)
	ElementToDiscardO: STRING; (*variabile utilizzata per visualizzare la "O" nella base che finisce nel cestino*)
	PistonBlackToDiscard: BOOL; (*variabile utilizzata per visualizzare il pistone nero nella base che finisce nel cestino*)
	PistonSilverToDiscard: BOOL; (*variabile utilizzata per visualizzare il pistone grigio nella base che finisce nel cestino*)


	DisplayText: STRING; (*variabile utilizzata per indicare passo-passo le istruzioni da compiere per un corretto funzionamento del sistema*)
	FillAllWarehouses:BOOL:=FALSE; (*variabile collegata al pulsante, che simula lo riempimento di tutti i magazzini della stazione di assemblaggio*)
	(**************************************************************************************************)


	(*VARIABILI DI SIMULAZIONE DIAGNOSTICA*)
	(*........................................................................................................................*)
	(*SENSORI*)
	AlignementRotaryTableWithPositioningsBlockHigh:BOOL;
	AlignementRotaryTableWithPositioningsBlockLow:BOOL;

	AvailableLoadForWorkingStationBlockHigh:BOOL;
	AvailableLoadForWorkingStationBlockLow:BOOL;

	AvailableLoadInControlPositioningBlockHigh:BOOL;
	AvailableLoadInControlPositioningBlockLow:BOOL;

	AvailableLoadInDrillingPositioningBlockHigh:BOOL;
	AvailableLoadInDrillingPositioningBlockLow:BOOL;

	InControlLoadInWrongPositionToBeDrilledBlockHigh: BOOL;
	InControlLoadInWrongPositionToBeDrilledBlockLow: BOOL;

	DrillingUnitUpBlockHigh: BOOL;
	DrillingUnitUpBlockLow: BOOL;

	DrillingUnitDownBlockHigh: BOOL;
	DrillingUnitDownBlockLow: BOOL;

	AvailableLoadForRobotBlockHigh: BOOL;
	AvailableLoadForRobotBlockLow: BOOL;

	RobotInInitialPositionBlockHigh: BOOL;
	RobotInInitialPositionBlockLow: BOOL;

	RobotInAssemblyUnitBlockHigh: BOOL;
	RobotInAssemblyUnitBlockLow: BOOL;

	RobotInPistonWarehouseBlockHigh: BOOL;
	RobotInPistonWarehouseBlockLow: BOOL;

	RobotInSpringWarehouseBlockHigh: BOOL;
	RobotInSpringWarehouseBlockLow: BOOL;

	RobotInCoverWarehouseBlockHigh: BOOL;
	RobotInCoverWarehouseBlockLow: BOOL;

	EmptyCoverHouseInAssemblyStationBlockHigh: BOOL;
	EmptyCoverHouseInAssemblyStationBlockLow: BOOL;

	ToExtractCoverInAssemblyStationInRetroactivePositionBlockHigh: BOOL;
	ToExtractCoverInAssemblyStationInRetroactivePositionBlockLow: BOOL;

	ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh: BOOL;
	ToExtractCoverInAssemblyStationInExtensivePositionBlockLow: BOOL;

	PistonSelectorIsOnTheRightBlockHigh: BOOL;
	PistonSelectorIsOnTheRightBlockLow: BOOL;

	PistonSelectorIsOnTheLeftBlockHigh: BOOL;
	PistonSelectorIsOnTheLeftBlockLow: BOOL;

	ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh: BOOL;
	ToExtractSpringInAssemblyStationInExtensivePositionBlockLow: BOOL;

	ToExtractSpringInAssemblyStationInRetroactivePositionBlockHigh: BOOL;
	ToExtractSpringInAssemblyStationInRetroactivePositionBlockLow: BOOL;

	(*  ATTUATORI  *)

	RotaryTableMotorBlockHigh: BOOL;
	RotaryTableMotorBlockLow: BOOL;

	ToLowerCylinderToInspectLoadBlockHigh: BOOL;
	ToLowerCylinderToInspectLoadBlockLow: BOOL;

	DrillingUnitActiveBlockHigh: BOOL;
	DrillingUnitActiveBlockLow: BOOL;

	ToLowerDrillingUnitBlockHigh: BOOL;
	ToLowerDrillingUnitBlockLow: BOOL;

	ToLiftDrillingUnitBlockHigh: BOOL;
	ToLiftDrillingUnitBlockLow: BOOL;

	BlockingCylinderForwardInDrillingPositioningBlockHigh: BOOL;
	BlockingCylinderForwardInDrillingPositioningBlockLow: BOOL;

	ExpellingLeverActiveBlockHigh: BOOL;
	ExpellingLeverActiveBlockLow: BOOL;

	ToExtractSpringInAssemblyStationBlockHigh: BOOL;
	ToExtractSpringInAssemblyStationBlockLow: BOOL;

	PistonSelectorGoOnTheRightBlockHigh: BOOL;
	PistonSelectorGoOnTheRightBlockLow: BOOL;

	PistonSelectorGoOnTheLeftBlockHigh: BOOL;
	PistonSelectorGoOnTheLeftBlockLow: BOOL;

	ToExtractCoverInAssemblyStationForwardBlockHigh: BOOL;
	ToExtractCoverInAssemblyStationForwardBlockLow: BOOL;

	BlockingCylinderForwardInAssemblyStationBlockHigh: BOOL;
	BlockingCylinderForwardInAssemblyStationBlockLow: BOOL;

	RobotTakeBlackLoadBlockHigh: BOOL;
	RobotTakeBlackLoadBlockLow: BOOL;

	RobotTakeRedSilverLoadBlockHigh: BOOL;
	RobotTakeRedSilverLoadBlockLow: BOOL;

	RobotTakeLoadToDiascardBlockHigh: BOOL;
	RobotTakeLoadToDiascardBlockLow: BOOL;

	RobotGoToInitialPositionBlockHigh: BOOL;
	RobotGoToInitialPositionBlockLow: BOOL;

	RobotGoToSpringHouseBlockHigh: BOOL;
	RobotGoToSpringHouseBlockLow: BOOL;

	RobotGoToPistonHouseBlockHigh: BOOL;
	RobotGoToPistonHouseBlockLow: BOOL;

	RobotGoToCoverHouseBlockHigh: BOOL;
	RobotGoToCoverHouseBlockLow: BOOL;

	RobotTakeCurrentLoadToAssemblyBlockHigh: BOOL;
	RobotTakeCurrentLoadToAssemblyBlockLow: BOOL;

	RobotEngineBlockLow: BOOL;


	(*VARIABILI DI RILEVAZIONE DIAGNOSTICA*)
(*--------------------------------------------------------------------------------------------------*)

	(*SENSORI*)

	AlignementRotaryTableWithPositioningsBlockedHigh:BOOL;
	AlignementRotaryTableWithPositioningsBlockedLow:BOOL;

	AvailableLoadForWorkingStationBlockedHigh:BOOL;
	AvailableLoadForWorkingStationBlockedLow:BOOL;

	AvailableLoadInControlPositioningBlockedHigh:BOOL;
	AvailableLoadInControlPositioningBlockedLow:BOOL;

	AvailableLoadInDrillingPositioningBlockedHigh:BOOL;
	AvailableLoadInDrillingPositioningBlockedLow:BOOL;

	InControlLoadInWrongPositionToBeDrilledBlockedHigh: BOOL;
	InControlLoadInWrongPositionToBeDrilledBlockedLow: BOOL;

	DrillingUnitUpBlockedHigh: BOOL;
	DrillingUnitUpBlockedLow: BOOL;

	DrillingUnitDownBlockedHigh: BOOL;
	DrillingUnitDownBlockedLow: BOOL;

	AvailableLoadForRobotBlockedHigh: BOOL;
	AvailableLoadForRobotBlockedLow: BOOL;

	RobotInInitialPositionBlockedHigh: BOOL;
	RobotInInitialPositionBlockedLow: BOOL;

	RobotInAssemblyUnitBlockedHigh: BOOL;
	RobotInAssemblyUnitBlockedLow: BOOL;

	RobotInPistonWarehouseBlockedHigh: BOOL;
	RobotInPistonWarehouseBlockedLow: BOOL;

	RobotInSpringWarehouseBlockedHigh: BOOL;
	RobotInSpringWarehouseBlockedLow: BOOL;

	RobotInCoverWarehouseBlockedHigh: BOOL;
	RobotInCoverWarehouseBlockedLow: BOOL;

	EmptyCoverHouseInAssemblyStationBlockedHigh: BOOL;
	EmptyCoverHouseInAssemblyStationBlockedLow: BOOL;

	ToExtractCoverInAssemblyStationInRetroactivePositionBlockedHigh: BOOL;
	ToExtractCoverInAssemblyStationInRetroactivePositionBlockedLow: BOOL;

	ToExtractCoverInAssemblyStationInExtensivePositionBlockedHigh: BOOL;
	ToExtractCoverInAssemblyStationInExtensivePositionBlockedLow: BOOL;

	PistonSelectorIsOnTheRightBlockedHigh: BOOL;
	PistonSelectorIsOnTheRightBlockedLow: BOOL;

	PistonSelectorIsOnTheLeftBlockedHigh: BOOL;
	PistonSelectorIsOnTheLeftBlockedLow: BOOL;

	ToExtractSpringInAssemblyStationInExtensivePositionBlockedHigh: BOOL;
	ToExtractSpringInAssemblyStationInExtensivePositionBlockedLow: BOOL;

	ToExtractSpringInAssemblyStationInRetroactivePositionBlockedHigh: BOOL;
	ToExtractSpringInAssemblyStationInRetroactivePositionBlockedLow: BOOL;


	(*  ATTUATORI  *)

	RotaryTableMotorBlockedHigh: BOOL;
	RotaryTableMotorBlockedLow: BOOL;

	ToLowerCylinderToInspectLoadBlockedHigh: BOOL;
	ToLowerCylinderToInspectLoadBlockedLow: BOOL;

	DrillingUnitActiveBlockedHigh: BOOL;
	DrillingUnitActiveBlockedLow: BOOL;

	ToLowerDrillingUnitBlockedHigh: BOOL;
	ToLowerDrillingUnitBlockedLow: BOOL;

	ToLiftDrillingUnitBlockedHigh: BOOL;
	ToLiftDrillingUnitBlockedLow: BOOL;

	BlockingCylinderForwardInDrillingPositioningBlockedHigh: BOOL;
	BlockingCylinderForwardInDrillingPositioningBlockedLow: BOOL;

	ExpellingLeverActiveBlockedHigh: BOOL;
	ExpellingLeverActiveBlockedLow: BOOL;

	ToExtractSpringInAssemblyStationBlockedHigh: BOOL;
	ToExtractSpringInAssemblyStationBlockedLow: BOOL;

	PistonSelectorGoOnTheRightBlockedHigh: BOOL;
	PistonSelectorGoOnTheRightBlockedLow: BOOL;

	PistonSelectorGoOnTheLeftBlockedHigh: BOOL;
	PistonSelectorGoOnTheLeftBlockedLow: BOOL;

	ToExtractCoverInAssemblyStationForwardBlockedHigh: BOOL;
	ToExtractCoverInAssemblyStationForwardBlockedLow: BOOL;

	BlockingCylinderForwardInAssemblyStationBlockedHigh: BOOL;
	BlockingCylinderForwardInAssemblyStationBlockedLow: BOOL;

	RobotTakeBlackLoadBlockedHigh: BOOL;
	RobotTakeBlackLoadBlockedLow: BOOL;

	RobotTakeRedSilverLoadBlockedHigh: BOOL;
	RobotTakeRedSilverLoadBlockedLow: BOOL;

	RobotTakeLoadToDiascardBlockedHigh: BOOL;
	RobotTakeLoadToDiascardBlockedLow: BOOL;

	RobotGoToInitialPositionBlockedHigh: BOOL;
	RobotGoToInitialPositionBlockedLow: BOOL;

	RobotGoToSpringHouseBlockedHigh: BOOL;
	RobotGoToSpringHouseBlockedLow: BOOL;

	RobotGoToPistonHouseBlockedHigh: BOOL;
	RobotGoToPistonHouseBlockedLow: BOOL;

	RobotGoToCoverHouseBlockedHigh: BOOL;
	RobotGoToCoverHouseBlockedLow: BOOL;

	RobotTakeCurrentLoadToAssemblyBlockedHigh: BOOL;
	RobotTakeCurrentLoadToAssemblyBlockedLow: BOOL;

	RobotEngineBlockedLow: BOOL;

	RobotProgramRunningBlockedLow: BOOL;

	EndVerification: BOOL;
	Colour: BOOL;
	View: BOOL;

(*Variabili di comunicazione nel main*)
	InitializationFinished: BOOL;
       ReadyLoadForRotaryTable:BOOL:=FALSE;
       ExpelledLoadFromVerification:BOOL:=FALSE;
	LoadInVerification: BOOL;
	VerificationFree: BOOL; (*Indica la fine delle operazioni della stazione di verifica*)
	VirtualSensorConnectionAlignementRotaryTable: BOOL;
      CommunicationLoadInCorrectPosition:BOOL;
	EndRotaryTable: BOOL:=FALSE;   (*Variabile che indica la fine della rotazione della tavola per il timer*)
	ColorCurrentLoad: STRING;
       BufferColor: ARRAY [1..5] OF STRING:='Empty','Empty','Empty','Empty','Empty';(*Posto1:Inizio giostra. Posto2:Stazione di controllo. Posto3:Stazione di foratura. Posto4:Stazione di espulsione. Posto5:Stazione robot*)
       BufferWrongPosition: ARRAY [1..4] OF STRING;(*Posto1:Stazione di controllo. Posto2:Stazione di foratura. Posto3:Stazione di espulsione. Posto4:Stazione robot*)
	RobotRemovedPiston: BOOL;
	ColorVisualization:ARRAY[1..5] OF DWORD:=16#FFFFFF,16#FFFFFF,16#FFFFFF,16#FFFFFF,16#FFFFFF;
	PositionVisualization:ARRAY[1..4] OF DWORD:=16#FFFFFF,16#FFFFFF,16#FFFFFF,16#FFFFFF;
	LoadTransportOK: BOOL;(*Booleana usata per indicare che non c'� stato un problema nel trasporto del pezzo da magazzino a verifica*)

(*Variabili per la visualizzazione della macchina astati*)
	IndicatorStateIdle: BOOL;
	IndicatorStateCheck: BOOL;
	IndicatorStateInit: BOOL;
	IndicatorStateReady: BOOL;
	IndicatorStateBusy: BOOL;
	IndicatorStateStop: BOOL;
	IndicatorStateFreeze: BOOL;
	IndicatorStateAlarm: BOOL;
	IndicatorStateSafe: BOOL;
	EndCheck: BOOL;
	EndAlarm: BOOL;
	EndSafe: BOOL;
       EndStop1:BOOL;
       EndStop2:BOOL;
	EndFreeze: BOOL;

(*Luci pulsantiera lampeggianti in emergenza*)

	MachineNOTOkForReset: BOOL:=TRUE;   (*Indica se la macchina � piena e tale da no potere fare il reset*)
      TextMachineEmptyForReset:STRING;
	LightMachineToBeEmpty: BOOL;
	ButtonMachineToBeEmptyInvisible:BOOL:=FALSE; (*Serve a rendere invisibile il pulsante di richiesta di svuotamento macchina*)
	TypeOfFault: STRING; (*Utilizzata nella visualizzazione per vedere il tipo di guasto rilevato*)
	FaultRemoved: BOOL; (*Pulsante utente per rimuovore l'allarme dovuto a guasti*)
	PossibleFault: BOOL;  (*Utilizzata dal Diagnoser per indicare se � stato rilevata una situazione di possibile guasto. La machina viene fermata in fase*)

(***************************************)
(*****TIMERS DI DIAGNOSTICA******)
(***************************************)

	ExtractionCyliderInActuation: TON; (*Timer cilindro distribuzione in estrazione*)
	ExtractionCyliderInDeactivation: TON; (*Timer cilindro distribuzione in retrazione*)
	ElevatorInUpMovement: TON;  (*Timer elevatore in salita*)
	ElevatorInDownMovement: TON;  (*Timer elevatore in discesa*)
	DiagRotaryTableActivation: TON;   (*Timer tavola rotante in attivazione*)
	DiagRotaryTableDeactivation: TON; (*Timer tavola rotante in disattivazione*)

(*Variabili per l'attivazione dei timer di diagnostica*)

	TimeDiagExtractCylinderInActuation: BOOL; (*Attivazione timer diagnostica estrazione cilindro*)
	TimeDiagExtractCylinderInDeactivation: BOOL; (*Attivazione timer diagnostica retrazione cilindro*)
	TimeDiagElevatorInUpMovement: BOOL; (*Attivazione timer diagnostica elevatore in salita*)
	TimeDiagElevatorInDownMovement: BOOL; (*Attivazione timer diagnostica elevatore in discesa*)
	TimeDiagRotaryTableDeactivation: BOOL;  (*Attivazione timer diagnostica tavola in disattivazione*)
	TimeDiagRotaryTableActivation: BOOL; (*Attivazione timer diagnostica tavola in attivazione*)

(**********************************************************************)
(*****STRINGE DI COMUNICAZIONE MALFUNZIONAMENTI******)
(**********************************************************************)

	ComunicationFaultElevator: STRING; (*Tipo di fault per l'elevatore*)
	ComunicationFaultRotaryTable:STRING; (*Tipo di fault per la tavola rotante*)
	ComunicationFaultExtractionCylinder:STRING; (*Tipo di fault per il cilindro di estrazione*)
       ComunicationFaultSensorPresenceVerification:STRING; (*Tipo di fault per il sensore di presenza in verifica*)

(**********************************************************************)
(*****VARIABILI DELL' INTERFACCIA GRAFICA DI ALLARME******)
(**********************************************************************)
	SensorCylinderExtractFault:BOOL; (*Guasto al finecorsa esteso*)
	SensorCylinderRetractFault:BOOL; (*Guasto al finecorsa retratto*)
	CylinderMovementFault:BOOL; (*Guasto all'attuatore cilindro*)
	SensorElevatorUpFault:BOOL; (*Guasto al finecorsa alto*)
	SensorElevatorDownFault:BOOL; (*Guasto al finecorsa basso*)
	ElevatorMovementFault:BOOL; (*Guasto all'attuatore elevatore*)
	SensorAlignementRotaryTableFault:BOOL; (*Guasto al sensore allineamento tavola*)
	RotaryTableMovementFault:BOOL; (*Guasto all'attuatore tavola*)
	SensorLoadInVerificationFault:BOOL; (*Guasto al sensore presenza in verifica*)
	LoadDroppedByTheCarrierFault:BOOL; (*Guasto pezzo caduto dal braccio rotante*)
	SensorElevatorDownPossibleFault:BOOL; (*Possibile guasto al finecorsa basso*)
END_VAR
                                                                                               '           [   , $ �           Testing_Global_Variables ��d	��d[     ��������        �  (* GDs - Actuators and Sensors*)
VAR_GLOBAL
	(*Elevator - DA_DF *)
     enable_Elevator : BOOL;
     disable_Elevator : BOOL;
     Elevator_enabled : BOOL;
     Elevator_disabled : BOOL;

	(*ExtractionCylinder - SA_SDF *)
     enable_ExtractionCylinder : BOOL;
     ExtractionCylinder_disabled : BOOL;

	(*AirCushion - SA_NF*)
     enable_AirCushion : BOOL;

	(* Pure sensors *)
	ReadyLoadForVerificationLogical:BOOL;
	MeasurementNotOkLogical:BOOL;
	ColourMeasurementLogical:BOOL;
END_VAR                                                                                                 I   |0|0 @?    @   Arial @        @           �����                               �      �   ���  �3 ���   � ���     
    @��  ���     @      DEFAULT             System      I   |0|0 @?    @   Arial @        @           �����                      )   HH':'mm':'ss @                             dd'-'MM'-'yyyy @       '          f   , 70�           Assembly_States ��d	��d      ��������        H  TYPE
  	Assembly_States: (
		Assembly_ready_to_initialize,
		Assembly_initializing,
		Assembly_ready_to_enable,
		BlockingCylinder_enabling,
		ItemsSupply_enabling,
		ItemsAssembling_enabling,
		Assembly_ready_to_disable,
		ItemsSupply_disabling,
		BlockingCylinder_disabling,
		ItemsAssembling_disabling );
END_TYPE             -   , � � �           Data_Handler ��d	��d      ��������        �   TYPE Data_Handler :
STRUCT
	ID:INT:=0;
	Colour:BOOL:=FALSE;
	Height:BOOL:=FALSE;
	Orientation:BOOL:=FALSE;
	Discard:BOOL:=FALSE;
END_STRUCT
END_TYPE             ;   , n � �           DisTest_States ��d	��d      ��������        �   TYPE DisTest_States :(
	DisTest_ready_to_initialize,
	DisTest_Initializing,
	DisTest_ready_to_run,
 	Distribution_enabling,
 	Distribution_disabling,
	Testing_enabling,
	Testing_waiting,
	Testing_disabling);
END_TYPE
             =   , �0 `           Distribution_States ��d	��d      ��������        �  TYPE
  	Distribution_States: (
		Distribution_ready_to_initialize,
		Distribution_initializing,
		Distribution_ready_to_enable,
		Rotary_disabling_no_piece,
		Cylinder_distribution_enabling,
		Rotary_enabling_no_piece,
		VacuumGenerator_enabling,
		VacuumGenerator_disabling,
		Distribution_ready_to_disable,
		Cylinder_distribution_disabling,
		Rotary_disabling_with_piece,
		ExpulsionAirVacuum_enabling,
		ExpulsionAirVacuum_disabling,
		Rotary_enabling_with_piece);
END_TYPE             O   , � � .           DrillingUnit_States ��d	��d      ��������        C  TYPE
   DrillingUnit_States : (
	Drilling_ready_to_initialize,
	Drilling_initializing,
	Drilling_ready_to_enable,
	Holding_of_workpiece_enabling,
	Drilling_enabling,
	Drill_Machine_moving_down,
	Drilling_ready_to_disable,
	Drill_Machine_moving_up,
	Drilling_disabling,
	Holding_of_workpiece_disabling);
END_TYPE             R   , �� W           GenericDevice_States ��d	��d      ��������        �   TYPE GenericDevice_States :(
			DeviceDisabledState,
			DeviceEnablePhaseState,
			DeviceEnabledState,
			DeviceDisablePhaseState);
END_TYPE             K   , h h w�           InspectionUnit_States ��d	��d      ��������        �   TYPE
   InspectionUnit_States: (
	Inspection_ready_to_initialize,
	Inspection_initializing,
	Inspection_ready_to_enable,
	Cylinder_enabling,
	Inspection_ready_to_disable,
	Cylinder_disabling);
END_TYPE             .   ,  ]b(           ItemsAssemblingUnit_States ��d	��d      ��������        �  TYPE ItemsAssemblingUnit_States : (
		ItemsAssemblingUnit_ready_to_initialize,
		ItemsAssemblingUnit_initializing,
		ItemsAssemblingUnit_ready_to_enable,
		TakePiston_enabling,
		TakePiston_disabling,
		MoveToAssembly_enabling_with_piston,
		MoveToAssembly_disabling_with_piston,
		TakeSpring_enabling,
		TakeSpring_disabling,
		MoveToAssembly_enabling_with_spring,
		MoveToAssembly_disabling_with_spring,
		TakeCover_enabling,
		TakeCover_disabling,
		MoveToAssembly_enabling_with_cover,
		MoveToAssembly_disabling_with_cover,
		ItemsAssemblingUnit_ready_to_disable,
		Take_finished_piece_enabling,
		Take_finished_piece_disabling);
END_TYPE             j   , t Y�           ItemsSupplyUnit_States ��d	��d      ��������        (  TYPE ItemsSupplyUnit_States : (
		ItemsSupplyUnit_ready_to_initialize,
		ItemsSupplyUnit_initializing,
		ItemsSupplyUnit_ready_to_enable,
		ItemsSupplyUnit_enabling,
		PistonSelector_disabling_for_RedSilver_piece,
		ItemsSupplyUnit_ready_to_disable,
		ItemsSupplyUnit_disabling);
END_TYPE             _   , 5/ �8           Main_States ��d	��d      ��������        �   TYPE Main_States : (
		Ready_to_initialize,
		Initializing,
		Ready_to_Run,
		Running,
		ImmediateStopping,
		OnPhaseStopping,
		Stopping);
END_TYPE             T   , � � �P           Making_States ��d	��d      ��������        �   TYPE Making_States:(
		M_Processing_ready_to_initialize,
		M_Processing_Initializing,
		M_Processing_ready_to_run,
		Processing_waiting_to_receive,
		Processing_enabling,
		Processing_waiting_to_expel,
		Processing_disabling);
END_TYPE             '   ,   z�           PickandPlace_States ��d	��d      ��������        m  TYPE
   PickandPlace_States :(
	PickandPlace_ready_to_initialize,
	PickandPlace_initializing,
	PickandPlace_ready_to_enable,
	Robot_to_initial_position_enabling,
	Robot_to_initial_position_disabling,
	PickandPlace_ready_to_disable,
	Discarding_wrong_oriented_redsilver_enabling,
	Discarding_wrong_oriented_redsilver_disabling,
	Discarding_wrong_oriented_black_enabling,
	Discarding_wrong_oriented_black_disabling,
	Taking_correctly_oriented_redsilver_enabling,
	Taking_correctly_oriented_redsilver_disabling,
	Taking_correctly_oriented_black_enabling,
	Taking_correctly_oriented_black_disabling);
END_TYPE             E   , * p ��           Processing_States ��d	��d      ��������        $  TYPE
	Processing_States : (
		Processing_ready_to_initialize,
		Processing_initializing,
		Processing_ready_to_enable,
		Rotary_enabling,
		Rotary_disabling,
		Units_enabling,
		Units_disabling,
		Processing_ready_to_disable,
		Expelling_enabling,
		Expelling_disabling);
END_TYPE             &   , � � �/           Robot_States ��d	��d      ��������        �   TYPE Robot_States:(
	Robot_ready_to_initialize,
	Robot_Initializing,
	Robot_ready_to_run,
	PickandPlace_enabling,
	PickandPlace_disabling,
	Assembly_enabling,
	Assembly_disabling);
END_TYPE             s   , , B ��           SignalManagement_States ��d	��d      ��������        �   TYPE SignalManagement_States :(
			INIT_SIGNALMANAGEMENT,
			START_CONFIGURATION,
			CONFIGURATION,
			START_GENERATION,
			GENERATION,
			RUN_SIGNALMANAGEMENT);
END_TYPE
             ,   , ��� lN           Subsystem_Handler ��d	��d      ��������        �   TYPE Subsystem_Handler :
STRUCT
	Initialize:BOOL:=FALSE;
	Enable:BOOL:=FALSE;
	Disable:BOOL:=FALSE;
	ImmediateStop:BOOL:=FALSE;
	OnPhaseStop:BOOL:=FALSE; (* Not used *)
END_STRUCT
END_TYPE             `   , ��  �k           System_Handler ��d	��d      ��������        }   TYPE System_Handler :
STRUCT
	Initialize:BOOL:=FALSE;
	Run:BOOL:=FALSE;
	ImmediateStop:BOOL:=FALSE;
END_STRUCT
END_TYPE             Z   ,   �            Testing_States ��d	��d      ��������        �  TYPE
	Testing_States: (
		Testing_ready_to_initialize,
		Testing_initializing,
		Testing_ready_to_enable,
		Elevator_enabling,
		Testing_waiting_to_disable,
		Elevator_disabling_NOT_OK,
		ExtractionCylinder_enabling_NOT_OK,
		ExtractionCylinder_disabling_NOT_OK,
		AirCushion_enabling_OK,
		ExtractionCylinder_enabling_OK,
		ExtractionCylinder_disabling_OK,
		AirCushion_disabling_OK,
		Elevator_disabling_OK);
END_TYPE             ' u   ,   h�           Assembly_PRG ��d	��d      ��������        �  PROGRAM Assembly_PRG
VAR
	OperationType : INT := INIT;
	state : Assembly_States := Assembly_ready_to_initialize;
END_VAR


(* Between ROBOT_Prg and ASSEMBLY_PRG *)
VAR_EXTERNAL
	Assembly_Handler:		Subsystem_Handler;
	Assembly_Data:			Data_Handler;
END_VAR

(* Between ASSEMBLY_Prg and ITEMSSUPPLY_Prg*)
VAR_EXTERNAL
	ItemsSupply_Handler : 		Subsystem_Handler;
	ItemsSupply_SubData:		Data_Handler;
END_VAR

(* Between ASSEMBLY_Prg and ITEMSASSEMBLING_Prg*)
VAR_EXTERNAL
	ItemsAssembling_Handler : 		Subsystem_Handler;
END_VAR


(* GDs - Instances and  Handler request*)
VAR
	BlockingCylinder : Generic_Device;
	BlockingCylinder_enable_request : BOOL;
	BlockingCylinder_disable_request : BOOL;
	BlockingCylinder_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(*BlockingCylinder - SA_NF*)
     enable_BlockingCylinder : BOOL;
     BlockingCylinder_ActuatorFault : BOOL;

	(* Pure sensors *)
	EmptyCoverHouseInAssemblyStation_Logical: BOOL;
END_VAR�  (*** FSM ***)
IF Assembly_Handler.ImmediateStop THEN
	ItemsSupply_Handler.ImmediateStop:=TRUE;
	ItemsAssembling_Handler.ImmediateStop:=TRUE;
ELSE
	ItemsSupply_Handler.ImmediateStop:=FALSE;
	ItemsAssembling_Handler.ImmediateStop:=FALSE;
	CASE state OF
	Assembly_ready_to_initialize:
		IF Assembly_Handler.Initialize THEN
			OperationType := INIT;
			ItemsSupply_Handler.Initialize := TRUE;
			ItemsAssembling_Handler.Initialize := TRUE;
	
			state := Assembly_initializing;
		END_IF;
	
	Assembly_initializing:
		IF (NOT BlockingCylinder_not_initialized AND NOT ItemsSupply_Handler.Initialize AND NOT ItemsAssembling_Handler.Initialize) THEN
			OperationType := RUN;
			Assembly_Handler.Initialize := FALSE;
	
			state := Assembly_ready_to_enable;
		END_IF;
	
	Assembly_ready_to_enable:
		IF Assembly_Handler.Enable THEN
			BlockingCylinder_enable_request := TRUE;
	
			state := BlockingCylinder_enabling;
		END_IF;
	
	BlockingCylinder_enabling:
		IF (NOT BlockingCylinder_enable_request AND NOT EmptyCoverHouseInAssemblyStation_Logical ) THEN
			ItemsSupply_SubData := Assembly_Data;
			ItemsSupply_Handler.Enable := TRUE;
	
			state := ItemsSupply_enabling;
		END_IF;
	
	ItemsSupply_enabling:
		IF NOT ItemsSupply_Handler.Enable THEN (*� uscito il correto pistone *) (* Black piece -> needs Silver piston*)  (* Red/Silver piece -> needs Black piston*)
			ItemsAssembling_Handler.Enable := TRUE; (*Faccio partire il robot perch� ho gi� i pistoni giusti da montare *)
	
			state := ItemsAssembling_enabling;
		END_IF;
	
	ItemsAssembling_enabling:
		IF NOT ItemsAssembling_Handler.Enable THEN
			Assembly_Handler.Enable := FALSE;
	
			state := Assembly_ready_to_disable;
		END_IF;
	
	Assembly_ready_to_disable:
		IF Assembly_Handler.Disable THEN
			ItemsSupply_Handler.Disable := TRUE; (*Pu� essere che il pistone sia gi� in Disablign ma gli altri no*)
	
	
			state := ItemsSupply_disabling;
		END_IF;
	
	ItemsSupply_disabling:
		IF NOT ItemsSupply_Handler.Disable THEN
			BlockingCylinder_disable_request := TRUE;
	
			state := BlockingCylinder_disabling;
		END_IF;
	
	BlockingCylinder_disabling:
		IF NOT BlockingCylinder_disable_request THEN
			ItemsAssembling_Handler.Disable := TRUE; (* Non fanno niente*)
	
			state := ItemsAssembling_disabling;
		END_IF;
	
	ItemsAssembling_disabling:
		IF NOT ItemsAssembling_Handler.Disable THEN
			Assembly_Handler.Disable := FALSE;
	
			state := Assembly_ready_to_enable;
		END_IF;
	

	END_CASE;

END_IF

(*** GENERIC DEVICES ***)
BlockingCylinder.DeviceOperation := OperationType;
BlockingCylinder.DeviceClock := TRUE;
BlockingCylinder.DeviceDiagnosticsEnabled := TRUE;
BlockingCylinder.DeviceEnablePreset := FALSE;
BlockingCylinder.DeviceEnableTime := BlockingCylinder_EnableTime;
BlockingCylinder.DeviceDisableTime := BlockingCylinder_DisableTime;
BlockingCylinder.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
BlockingCylinder(DeviceEnableRequest := BlockingCylinder_enable_request, DeviceDisableRequest := BlockingCylinder_disable_request );
enable_BlockingCylinder := BlockingCylinder.EnableDevice;
BlockingCylinder_not_initialized:=BlockingCylinder.DeviceNotInitialized;
BlockingCylinder_ActuatorFault := BlockingCylinder.DeviceActuatorFault;               >   , � ���~           Bridge_Assembly_GDs ��d	��d      ��������        )   PROGRAM Bridge_Assembly_GDs
VAR
END_VARX  (*BlockingCylinder - SA_NF*)
BlockingCylinderForwardInAssemblyStation:= NOT enable_BlockingCylinder; (*COMANDO DI ATTUAZIONE: cilindro di boccaggio, SE TRUE SI SBLOCCA il pezzo *)

(* Pure sensors *)
EmptyCoverHouseInAssemblyStation_Logical:=EmptyCoverHouseInAssemblyStation; (*SENSORE: segnala se sono presenti dei coperchi nel magazzino*)               7   , n � �           Bridge_Distribution_GDs ��d	��d      ��������        -   PROGRAM Bridge_Distribution_GDs
VAR
END_VAR  (* GDs - Actuators and Sensors*)
(* Cylinder - SA_DF *)
Cylinder_enabled:=CylinderExtractionLoadInExtensivePosition; (*sensore cilindro di estrazione in posizione estratta*)
Cylinder_disabled:=CylinderExtractionLoadInRetroactivePosition; (*sensore cilindro di estrazione in posizione retratta*)
CylinderExtractsLoadFromWarehouse:=enable_Cylinder; (*comando di estrazione pezzo dal magazzino *)

(* RotaryMaker - DA_DF *)
RotaryMaker_enabled := RotaryMakerInPositionWarehouse; (*sensore di finecorsa rotary maker in posizione magazzino pezzi *)
RotaryMaker_disabled := RotaryMakerInPositionVerification;  (*sensore di finecorsa rotary maker in stazione di verifica *)
RotaryMakerVsWarehouse:=enable_RotaryMaker; (*comando rotary maker verso magazzino pezzi *)
RotaryMakerVsVerification:= disable_RotaryMaker; (*comando rotary maker verso stazione di verifica *)

(* VacuumGenerator - SA_SAF *)
VacuumGenerator:=enable_VacuumGenerator; (*comando generatore di vuoto *)
VacuumGenerator_enabled:=VacuumGeneratorOK;(*sensore di vuoto*)

(* ExpulsionAirVacuum - SA_NF *)
ExpulsionAirVacuum:=enable_ExpulsionAirVacuum; (*comando getto d'aria per espulsione *)

(* GDs - Sensors - Logical variables *)
EmptyWarehouse_Logical:=EmptyWarehouse; (*sensore magazzino pezzi vuoto*)               8   , +             Bridge_Drilling_GDs ��d	��d      ��������        )   PROGRAM Bridge_Drilling_GDs
VAR
END_VAR�  (* GDs - Actuators and Sensors*)
(* Drill_Machine - DA_DF *)
ToLowerDrillingUnit := enable_Drill_Machine;  (*COMANDO DI ATTUAZIONE:  l'unit� di foratura viene abbassata *)
ToLiftDrillingUnit := disable_Drill_Machine; (*COMANDO DI ATTUAZIONE: l'unit� di foratura viene sollevata *)
Drill_Machine_enabled := DrillingUnitDown;  (* SENSORE: unit� di foratura sulla base del pezzo *)
Drill_Machine_disabled := DrillingUnitUp; (* SENSORE: unit� di foratura in posizione sollevata*)

(* Drilling - SA_NF *)
DrillingUnitActive := enable_Drilling; (* COMANDO DI ATTUAZIONE: unit� di foratura ruota in senso orario *)

(* Holding - SA_NF *)
BlockingCylinderForwardInDrillingPositioning := enable_Holding; (*COMANDO DI ATTUAZIONE: pistone di bloccaggio pezzo*)

(* Pure sensors *)
AvailableLoadInDrillingPositioning_Logical := AvailableLoadInDrillingPositioning; (*SENSORE DI PRESENZA: load presente nel modulo di foratura*)               M   , � � w�           Bridge_Inspection_GSs ��d	��d      ��������           PROGRAM Bridge_Inspection_GSs  (* GDs - Actuators and Sensors*)
(* CylinderToInspect - SA_NF *)
ToLowerCylinderToInspectLoad := enable_CylinderToInspect; (* COMANDO DI ATTUAZIONE: L'unit� di controllo viene abbassata *)

(* GDs - Sensors - Logical variables *)
AvailableLoadInControlPositionLogical := AvailableLoadInControlPositioning; (*SENSORE DI PRESENZA: load presente in postazione di controllo*)
InControlLoadInWrongPositionToBeDrilledLogical := InControlLoadInWrongPositionToBeDrilled; (*SENSORE per il rilevamento del corretto orientamento della base*)

               g   , 4  hZ           Bridge_ItemsAssembling_GDs ��d	��d      ��������        $   PROGRAM Bridge_ItemsAssembling_GDs
�  (* Actuators bridge - Output bridge*)
IF RobotTakeCurrentLoadToAssembly_Logical THEN (*R9*)
	Bit3Output := TRUE;
	Bit2Output := FALSE;
	Bit1Output := FALSE;
	Bit0Output := TRUE;
END_IF;

IF RobotGoToPistonHouse_Logical THEN (*R7*)
	Bit3Output := FALSE;
	Bit2Output := TRUE;
	Bit1Output := TRUE;
	Bit0Output := TRUE;
END_IF;

IF RobotGoToSpringHouse_Logical THEN (*R6*)
	Bit3Output := FALSE;
	Bit2Output := TRUE;
	Bit1Output := TRUE;
	Bit0Output := FALSE;
END_IF;

IF RobotGoToCoverHouse_Logical THEN (*R7*)
	Bit3Output := TRUE;
	Bit2Output := FALSE;
	Bit1Output := FALSE;
	Bit0Output := FALSE;
END_IF;

IF RobotGoToInitialPosition_Logical_A THEN (*R1*)
	Bit3Output := FALSE;
	Bit2Output := FALSE;
	Bit1Output := FALSE;
	Bit0Output := TRUE;
END_IF;

(*Sensors bridge - Input Bridge*)
(* Feedback is active only when actuator is active, to have no fault in disable request*)
RobotInAssemblyUnit_Logical_A := 	(NOT Bit0Input AND NOT Bit1Input AND NOT Bit2Input AND Bit3Input)  AND 	(RobotTakeCurrentLoadToAssembly_Logical);
RobotInPistonWarehouse_Logical := 	(Bit0Input AND  NOT Bit1Input AND NOT Bit2Input AND NOT Bit3Input) AND 	(RobotGoToPistonHouse_Logical);
RobotInSpringWarehouse_Logical := 	(NOT Bit0Input AND Bit1Input AND NOT Bit2Input AND NOT Bit3Input)  AND 	(RobotGoToSpringHouse_Logical);
RobotInCoverWarehouse_Logical := 	(NOT Bit0Input AND NOT Bit1Input AND Bit2Input AND NOT Bit3Input)  AND 	(RobotGoToCoverHouse_Logical);
RobotInInitialPosition_Logical_A := 	(Bit0Input AND Bit1Input AND NOT Bit2Input AND NOT Bit3Input)         AND 	(RobotGoToInitialPosition_Logical_A);

(*R6*)
RobotGoToSpringHouse_Logical := enable_R_TakeSpring;
R_TakeSpring_enabled := RobotInSpringWarehouse_Logical;

(*R7*)
RobotGoToPistonHouse_Logical := enable_R_TakePiston;
R_TakePiston_enabled := RobotInPistonWarehouse_Logical;

(*R8*)
RobotGoToCoverHouse_Logical := enable_R_TakeCover;
R_TakeCover_enabled := RobotInCoverWarehouse_Logical;

(*R9*)
RobotTakeCurrentLoadToAssembly_Logical := enable_R_AssemblyPosition;
R_AssemblyPosition_enabled := RobotInAssemblyUnit_Logical_A;

(*R1*)
RobotGoToInitialPosition_Logical_A := enable_R_Initial_position_A;
R_Initial_position_enabled_A := RobotInInitialPosition_Logical_A;
               l   , �  `�           Bridge_ItemsSupply_GDs ��d	��d      ��������        ,   PROGRAM Bridge_ItemsSupply_GDs
VAR
END_VAR�  (*PistonSelector - DA-DF*)
PistonSelectorGoOnTheRight := enable_PistonSelector; (*COMANDO DI ATTUAZIONE: magazzino pistoni ruota verso destra (preleva pistone NERO)*)
PistonSelectorGoOnTheLeft:= disable_PistonSelector; (*COMANDO DI ATTUAZIONE: magazzino pistoni ruota verso sinistra(preleva pistone GRIGIO)*)
PistonSelector_enabled:=PistonSelectorIsOnTheRight; (* SENSORE: magazzino pistoni ruotato completamente a destra *)
PistonSelector_disabled:=PistonSelectorIsOnTheLeft; (* SENSORE: magazzino pistoni ruotato completamente a sinistra *)

(*ExtractSpring - SA-DF*)
ToExtractSpringInAssemblyStation:= NOT enable_ExtractSpring; (* COMANDO DI ATTUAZIONE del cilindro che preleva la molla dal magazzino *)
ExtractSpring_enabled:=ToExtractSpringInAssemblyStationInExtensivePosition; (* SENSORE di fine corsa che indica che il cilindro � in posizione estesa(di riposo)*)
ExtractSpring_disabled:=ToExtractSpringInAssemblyStationInRetroactivePosition; (*SENSORE di fine corsa che indica che il cilindro � in posizione ritratta*)

(*ExtractCover - SA_DF*)
ToExtractCoverInAssemblyStationForward := enable_ExtractCover; (*COMANDO DI ATTUAZIONE: cilindro estrae cover *)
ExtractCover_disabled:=ToExtractCoverInAssemblyStationInRetroactivePosition; (*SENSORE:  cilindro di estrazione in posizione retratta *)
ExtractCover_enabled:=ToExtractCoverInAssemblyStationInExtensivePosition; (*SENSORE: cilindro di estrazione in posizione estesa *)               )   , � � R�           Bridge_PickandPlace_GDs ��d	��d      ��������        [   PROGRAM Bridge_PickandPlace_GDs

(*
VAR
	RobotNullCommand : BOOL := FALSE;
END_VAR
*)	  (*Actuators bridge - Output bridge*)
IF RobotGoToInitialPosition_Logical THEN (*R1*)
	Bit3Output := FALSE;
	Bit2Output := FALSE;
	Bit1Output := FALSE;
	Bit0Output := TRUE;
ELSIF RobotTakeBlackLoad_Logical THEN (* R2*)
	Bit3Output := FALSE;
	Bit2Output := FALSE;
	Bit1Output := TRUE;
	Bit0Output := FALSE;
ELSIF RobotTakeRedSilverLoad_Logical THEN
	Bit3Output := FALSE;
	Bit2Output := FALSE;
	Bit1Output := TRUE;
	Bit0Output := TRUE;
ELSIF RobotTakeLoadToDiascardBlack_Logical THEN
	Bit3Output := FALSE;
	Bit2Output := TRUE;
	Bit1Output := FALSE;
	Bit0Output := TRUE;
ELSIF RobotTakeLoadToDiascardRedSilver_Logical THEN
	Bit3Output := FALSE;
	Bit2Output := TRUE;
	Bit1Output := FALSE;
	Bit0Output := TRUE;
ELSE
	Bit3Output := FALSE;
	Bit2Output := FALSE;
	Bit1Output := FALSE;
	Bit0Output := FALSE;
END_IF

(*IF RobotTakeLoadToDiascardRedSilver THEN
	Bit3Output := FALSE;
	Bit2Output := TRUE;
	Bit1Output := FALSE;
	Bit0Output := FALSE;
END_IF*)

(*Sensors bridge - Input Bridge*)
(* Feedback is active only when actuator is active, to have no fault in disable request*)
RobotInInitialPosition_Logical := (Bit0Input AND Bit1Input AND NOT Bit2Input AND NOT Bit3Input) 		AND (RobotGoToInitialPosition_Logical OR RobotTakeLoadToDiascardBlack_Logical OR RobotTakeLoadToDiascardRedSilver_Logical);
RobotInAssemblyUnit_Logical := (NOT Bit0Input AND NOT Bit1Input AND NOT Bit2Input AND Bit3Input) 	AND (RobotTakeBlackLoad_Logical OR RobotTakeRedSilverLoad_Logical);


(*R1*)
RobotGoToInitialPosition_Logical := enable_R_Initial_position;
R_Initial_position_enabled  := RobotInInitialPosition_Logical;

(*R2*)
RobotTakeBlackLoad_Logical := enable_R_Take_black_piece;
R_Take_black_piece_enabled := RobotInAssemblyUnit_Logical;

(*R3*)
RobotTakeRedSilverLoad_Logical := enable_R_Take_redsilver_piece;
R_Take_redsilver_piece_enabled := RobotInAssemblyUnit_Logical;

(*R4*)
RobotTakeLoadToDiascardBlack_Logical := enable_R_Take_black_upsidedown_piece;
R_Take_black_upsidedown_piece_enabled := RobotInInitialPosition_Logical;

(*R5*)
RobotTakeLoadToDiascardRedSilver_Logical := enable_R_Take_redsilver_upsidedown_piece;
R_Take_redsilver_upsidedown_piece_enabled := RobotInInitialPosition_Logical;


(* Pure Sensor *)
AvailableLoadForRobot_Logical_PP := AvailableLoadForRobot;

               B   , � � ��           Bridge_Processing_GDs ��d	��d      ��������           PROGRAM Bridge_Processing_GDs�  (* GDs - Actuators and Sensors*)
(* RotaryTable - DANR_DF *)
RotaryTableMotor := (enable_RotaryTable OR disable_RotaryTable);  (* COMANDO DI ATTUAZIONE: tavola rotante *)
RotaryTable_enabled := NOT AlignementRotaryTableWithPositionings; (* Identifica quando disallineata*)
RotaryTable_disabled := AlignementRotaryTableWithPositionings; (* SENSORE: tavola allineata con le postazioni *)

(* ExpellingLever - SA_NF *)
ExpellingLeverActive := enable_ExpellingLever; (*COMANDO DI ATTUAZIONE:  leva espelle pezzo *)

(* GDs - Sensors - Logical variables *)
AvailableLoadForWorkingStation_Logical := AvailableLoadForWorkingStation; (* SENSORE DI PRESENZA: load presente nella prima stazione della giostra*)               U   , f � ��           Bridge_Testing_GDs ��d	��d      ��������        (   PROGRAM Bridge_Testing_GDs
VAR
END_VARK  (* GDs - Actuators and Sensors*)
(*Elevator - DA_DF *)
Elevator_enabled :=CylinderUpToMeasureLoad;  (*SENSORE: sensore finecorsa modulo di sollevamento in alto *)
Elevator_disabled := CylinderDownToMeasureLoad; (*SENSORE: sensore finecorsa modulo di sollevamento in basso *)
ToLiftCylinderToMeasureLoad:=enable_Elevator; (*COMANDO DI ATTUAZIONE: comando modulo di sollevamento verso l'alto *)
ToLowerCylinderToMeasureLoad:= disable_Elevator; (*COMANDO DI ATTUAZIONE: comando modulo di sollevamento verso il basso *)

(*ExtractionCylinder - SA_SDF *)
ToExtendCylinderOfExtractionVsGuide:=enable_ExtractionCylinder;  (*COMANDO DI ATTUAZIONE: comando espulsione pezzo *)
ExtractionCylinder_disabled:=CylinderOfExtractionInRetroactivePosition; (*SENSORE: sensore finecorsa cilindro di estrazione in posizione retratta *)

(*AirCushion - SA_NF*)
AirCushion:=enable_AirCushion; (*COMANDO DI ATTUAZIONE: comando cuscinetto d'aria*)

(* Pure sensors *)
ReadyLoadForVerificationLogical:=ReadyLoadForVerification; (*SENSORE DI PRESENZA: sensore presenza pezzo alla base della stazione di verifica *)
MeasurementNotOkLogical:=MeasurementNotOk; (*SENSORE: sensore di misuarazione altezza. Uscita del misuratore: 1 pezzo alto, 0 pezzo basso*)
ColourMeasurementLogical := ColourMeasurement; (*sensore di rilevazione colore: 0 nero, 1 rosso/metallico *)               n   , � ts           Buttons ��d	��d      ��������        �  PROGRAM Buttons (* Input bridge *)
(*Input -Phisical*)
VAR_EXTERNAL
	Start : BOOL;
	Reset : BOOL;
	Stop: BOOL := TRUE;
	FreezeStopPuls:BOOL:=TRUE;
END_VAR

(*Input -HMI Panel*)
VAR_EXTERNAL
	Init_HMI : BOOL;
	Start_HMI : BOOL;
	Stop_HMI : BOOL;
	Reset_HMI : BOOL;
	OnPhaseStop_HMI: BOOL;
	ImmediateStop_HMI:BOOL;
END_VAR

(*Input - Signal*)
VAR_EXTERNAL
	ResetSignalsEnable : BOOL;

	OnPhaseStop_Signal : BOOL;
	ImmediateStop_Signal : BOOL;
END_VAR

(* Local - Buttons *)
VAR
	Start_Button : BOOL;
	Reset_Button : BOOL;
	OnPhaseStop_Button: BOOL;
	ImmediateStop_Button:BOOL;
END_VAR

(*Output - Logical*)
VAR_EXTERNAL
	Init_Logical : BOOL;
	Start_Logical : BOOL;
	Reset_Logical : BOOL;
	Stop_Logical : BOOL;
	OnPhaseStop_Logical: BOOL;
	ImmediateStop_Logical:BOOL;
END_VAR

(*Filters*)
VAR
	Filter_Start : Signal_Filter;
	Filter_Reset : Signal_Filter;
	Filter_OnPhaseStop : Signal_Filter;
	Filter_ImmediateStop : Signal_Filter;
END_VAR	  (* Button filtered *)
Filter_Start(Signal := Start);
Start_Button:= Filter_Start.DelayedSignal;

Filter_Reset(Signal := Reset);
Reset_Button:= Filter_Reset.DelayedSignal;

Filter_OnPhaseStop(Signal := NOT Stop);
OnPhaseStop_Button:= Filter_OnPhaseStop.DelayedSignal;

Filter_ImmediateStop(Signal := NOT FreezeStopPuls);
ImmediateStop_Button:= Filter_ImmediateStop.DelayedSignal;


(* Logical processed *)
Init_Logical 				:= Init_HMI;
Start_Logical 			:= Start_Button OR Start_HMI;
Reset_Logical			:= (Reset_Button OR Reset_HMI) AND ResetSignalsEnable;
Stop_Logical			:= Stop_HMI;
OnPhaseStop_Logical 	:= OnPhaseStop_Button OR OnPhaseStop_Signal OR OnPhaseStop_HMI;
ImmediateStop_Logical	:= ImmediateStop_Button OR ImmediateStop_Signal OR ImmediateStop_HMI;               9   , ���:O           Distribution_PRG ��d	��d      ��������        _  PROGRAM Distribution_PRG
(* Da ELIMINARE*)
VAR
	OperationType : INT := INIT;
END_VAR

VAR
	 state: Distribution_States := Distribution_ready_to_initialize;
END_VAR

VAR
	counter:INT:=0;
END_VAR

(* Handler - Between MACHINE_Prg and DISTRIBUTION_Prg *)
VAR_EXTERNAL
	Distribution_Handler:Subsystem_Handler;
	Distribution_Data:Data_Handler;
END_VAR

(* GDs - Instances and  Handler request*)
VAR
	Cylinder : Generic_Device;
	Cylinder_enable_request : BOOL;
	Cylinder_disable_request : BOOL;
	Cylinder_not_initialized : BOOL;

	RotaryMaker : Generic_Device;
	RotaryMaker_enable_request : BOOL;
	RotaryMaker_disable_request : BOOL;
	RotaryMaker_not_initialized : BOOL;

	VacuumGenerator : Generic_Device;
	VacuumGenerator_enable_request : BOOL;
	VacuumGenerator_disable_request: BOOL;
	VacuumGenerator_not_initialized : BOOL;

	ExpulsionAirVacuum : Generic_Device;
	ExpulsionAirVacuum_enable_request : BOOL;
	ExpulsionAirVacuum_disable_request : BOOL;
	ExpulsionAirVacuum_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(* Cylinder - SA_DF *)
     enable_Cylinder : BOOL;
     Cylinder_enabled : BOOL;
     Cylinder_disabled : BOOL;
     Cylinder_EnabledSensorFault : BOOL;
     Cylinder_DisabledSensorFault : BOOL;
     Cylinder_fault :BOOL;
     Cylinder_ActuatorFault : BOOL;

	(* RotaryMaker - DA_DF *)
     enable_RotaryMaker : BOOL;
     disable_RotaryMaker : BOOL;
     RotaryMaker_enabled : BOOL;
     RotaryMaker_disabled : BOOL;
     RotaryMaker_EnabledSensorFault : BOOL;
     RotaryMaker_DisabledSensorFault : BOOL;
     RotaryMaker_fault :BOOL;
     RotaryMaker_ActuatorFault : BOOL;

	(* VacuumGenerator - SA_SAF *)
     enable_VacuumGenerator : BOOL;
     VacuumGenerator_enabled : BOOL;
     VacuumGenerator_EnabledSensorFault : BOOL;
     VacuumGenerator_fault :BOOL;
     VacuumGenerator_ActuatorFault : BOOL;

	(* GDs - Sensors - Logical variables *)
     enable_ExpulsionAirVacuum : BOOL;
     ExpulsionAirVacuum_ActuatorFault : BOOL;

	(* Pure sensors *)
	EmptyWarehouse_Logical:BOOL:=TRUE;
END_VAR]  (*** FSM ***)
IF NOT Distribution_Handler.ImmediateStop THEN

	CASE state OF
	
	Distribution_ready_to_initialize:
	   IF DIstribution_Handler.Initialize THEN
	       OperationType := INIT;
	
	       state := Distribution_initializing;
	   END_IF;
	
	Distribution_initializing:
	   IF (NOT Cylinder_not_initialized AND NOT RotaryMaker_not_initialized AND NOT VacuumGenerator_not_initialized AND NOT  ExpulsionAirVacuum_not_initialized) THEN
	       OperationType := RUN;
		DIstribution_Handler.Initialize := FALSE;
	
	       state := Distribution_ready_to_enable;
	   END_IF;
	
	Distribution_ready_to_enable:
	   IF (DIstribution_Handler.Enable AND NOT EmptyWarehouse_Logical) THEN
	       RotaryMaker_disable_request := TRUE;
	
	       state := Rotary_disabling_no_piece;
	   END_IF;
	
	Rotary_disabling_no_piece:
	   IF NOT RotaryMaker_disable_request THEN
	       Cylinder_enable_request := TRUE;
	
	       state := Cylinder_distribution_enabling;
	   END_IF;
	
	Cylinder_distribution_enabling:
	   IF NOT Cylinder_enable_request THEN
	       RotaryMaker_enable_request := TRUE;
	
	       state := Rotary_enabling_no_piece;
	   END_IF;
	
	Rotary_enabling_no_piece:
	   IF NOT RotaryMaker_enable_request THEN
	       VacuumGenerator_enable_request := TRUE;
	
	       state := VacuumGenerator_enabling;
	   END_IF;
	
	VacuumGenerator_enabling:
	   IF NOT VacuumGenerator_enable_request   THEN
	       VacuumGenerator_disable_request := TRUE;
	
	       state := VacuumGenerator_disabling;
	   END_IF;
	
	VacuumGenerator_disabling:
	   IF NOT VacuumGenerator_disable_request THEN
		counter:=counter+1;
		Distribution_Data.ID:=counter;
		Distribution_Handler.Enable:=FALSE;
	
	       state := Distribution_ready_to_disable;
	   END_IF;
	
	Distribution_ready_to_disable:
	   IF (Distribution_Handler.Disable) THEN
	       Cylinder_disable_request := TRUE;
	
	       state := Cylinder_distribution_disabling;
	   END_IF;
	
	Cylinder_distribution_disabling:
	   IF NOT Cylinder_disable_request THEN
	       RotaryMaker_disable_request := TRUE;
	
	       state := Rotary_disabling_with_piece;
	   END_IF;
	
	Rotary_disabling_with_piece:
	   IF NOT RotaryMaker_disable_request THEN
	       ExpulsionAirVacuum_enable_request := TRUE;
	
	       state := ExpulsionAirVacuum_enabling;
	   END_IF;
	
	ExpulsionAirVacuum_enabling:
	   IF NOT ExpulsionAirVacuum_enable_request THEN
	       ExpulsionAirVacuum_disable_request := TRUE;
	
	       state := ExpulsionAirVacuum_disabling;
	   END_IF;
	
	ExpulsionAirVacuum_disabling:
	   IF NOT ExpulsionAirVacuum_disable_request THEN
	       RotaryMaker_enable_request := TRUE;
	
	       state := Rotary_enabling_with_piece;
	   END_IF;
	
	Rotary_enabling_with_piece:
	   IF NOT RotaryMaker_enable_request THEN
		Distribution_Handler.Disable:=FALSE;
	
	       state := Distribution_ready_to_enable;
	   END_IF;
	END_CASE;


END_IF (* Dell'immediate_stop *)


(*** GENERIC DEVICES ***)
Cylinder.DeviceOperation := OperationType;
Cylinder.DeviceClock := TRUE;
Cylinder.DeviceDiagnosticsEnabled := TRUE;
Cylinder.DeviceEnablePreset := FALSE;
Cylinder.DeviceEnabledSensor := Cylinder_enabled;
Cylinder.DeviceDisabledSensor := Cylinder_disabled;
Cylinder.DeviceEnableTime := Cylinder_EnableTime;
Cylinder.DeviceDisableTime := Cylinder_DisableTime;
Cylinder.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
Cylinder(DeviceEnableRequest := Cylinder_enable_request, DeviceDisableRequest := Cylinder_disable_request );
enable_Cylinder := Cylinder.EnableDevice;
Cylinder_not_initialized:=Cylinder.DeviceNotInitialized;
Cylinder_ActuatorFault := Cylinder.DeviceActuatorFault;
Cylinder_EnabledSensorFault := Cylinder.DeviceEnabledSensorFault;
Cylinder_DisabledSensorFault := Cylinder.DeviceDisabledSensorFault;
Cylinder_fault := Cylinder.DeviceFault;

RotaryMaker.DeviceOperation := OperationType;
RotaryMaker.DeviceClock := TRUE;
RotaryMaker.DeviceDiagnosticsEnabled := TRUE;
RotaryMaker.DeviceEnablePreset := TRUE;
RotaryMaker.DeviceEnabledSensor := RotaryMaker_enabled;
RotaryMaker.DeviceDisabledSensor := RotaryMaker_disabled;
RotaryMaker.DeviceEnableTime := RotaryMaker_EnableTime;
RotaryMaker.DeviceDisableTime := RotaryMaker_DisableTime;
RotaryMaker.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_DOUBLE_ACTUATION;
RotaryMaker(DeviceEnableRequest := RotaryMaker_enable_request, DeviceDisableRequest := RotaryMaker_disable_request );
enable_RotaryMaker := RotaryMaker.EnableDevice;
RotaryMaker_not_initialized:=RotaryMaker.DeviceNotInitialized;
disable_RotaryMaker := RotaryMaker.DisableDevice;
RotaryMaker_ActuatorFault := RotaryMaker.DeviceActuatorFault;
RotaryMaker_EnabledSensorFault := RotaryMaker.DeviceEnabledSensorFault;
RotaryMaker_DisabledSensorFault := RotaryMaker.DeviceDisabledSensorFault;
RotaryMaker_fault := RotaryMaker.DeviceFault;

VacuumGenerator.DeviceOperation := OperationType;
VacuumGenerator.DeviceClock := TRUE;
VacuumGenerator.DeviceDiagnosticsEnabled := TRUE;
VacuumGenerator.DeviceEnablePreset := FALSE;
VacuumGenerator.DeviceEnabledSensor := VacuumGenerator_enabled;
VacuumGenerator.DeviceEnableTime := VacuumGenerator_EnableTime;
VacuumGenerator.DeviceDisableTime := VacuumGenerator_DisableTime;
VacuumGenerator.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
VacuumGenerator(DeviceEnableRequest := VacuumGenerator_enable_request, DeviceDisableRequest := VacuumGenerator_disable_request );
enable_VacuumGenerator := VacuumGenerator.EnableDevice;
VacuumGenerator_not_initialized:=VacuumGenerator.DeviceNotInitialized;
VacuumGenerator_ActuatorFault := VacuumGenerator.DeviceActuatorFault;
VacuumGenerator_EnabledSensorFault := VacuumGenerator.DeviceEnabledSensorFault;
VacuumGenerator_fault := VacuumGenerator.DeviceFault;

ExpulsionAirVacuum.DeviceOperation := OperationType;
ExpulsionAirVacuum.DeviceClock := TRUE;
ExpulsionAirVacuum.DeviceDiagnosticsEnabled := TRUE;
ExpulsionAirVacuum.DeviceEnablePreset := FALSE;
ExpulsionAirVacuum.DeviceEnableTime := ExpulsionAirVacuum_EnableTime;
ExpulsionAirVacuum.DeviceDisableTime := ExpulsionAirVacuum_DisableTime;
ExpulsionAirVacuum.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
ExpulsionAirVacuum(DeviceEnableRequest := ExpulsionAirVacuum_enable_request, DeviceDisableRequest := ExpulsionAirVacuum_disable_request );
enable_ExpulsionAirVacuum := ExpulsionAirVacuum.EnableDevice;
ExpulsionAirVacuum_not_initialized:=ExpulsionAirVacuum.DeviceNotInitialized;
ExpulsionAirVacuum_ActuatorFault := ExpulsionAirVacuum.DeviceActuatorFault;
               C   , ��]�           DistTest_PRG ��d	��d      ��������          PROGRAM DistTest_PRG
(* Memory array to manage data memory and its operation *)
VAR_EXTERNAL
    	Memory_Data: ARRAY [1..8] OF Data_Handler;
END_VAR

(* Arrey's index *)
VAR_EXTERNAL CONSTANT
	Distribution_index:		UINT:=1;
	Testing_index:			UINT:=2;
	Rotary_index:			UINT:=3;
	Inspection_index:			UINT:=4;
	Drilling_index:			UINT:=5;
	Expelling_index:			UINT:=6;
	PickandPlace_index:		UINT:=7;
	Supply_index:			UINT:=8;
END_VAR



(* Between MAIN_Prg and DISTTEST_Prg *)
VAR_EXTERNAL
	DistTest_Handler : 	System_Handler;
END_VAR

(* Between DISTTEST_Prg and DISTRIBUTION_Prg *)
VAR_EXTERNAL
	Distribution_Handler:		Subsystem_Handler;
	Distribution_Data:		Data_Handler;
END_VAR

(* Between DISTTEST_Prg and TESTING_Prg *)
VAR_EXTERNAL
	Testing_Handler:Subsystem_Handler;
	Testing_Data:Data_Handler;
END_VAR

(* Between DISTTEST_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Testing_ready_to_send :BOOL:= FALSE;
END_VAR

(* Between MAKING_Prg and DISTTEST_Prg *)
VAR_EXTERNAL
      Processing_ready_to_receive :BOOL:= FALSE;
END_VAR




VAR
	state_DisTest: DisTest_States := DisTest_ready_to_initialize;
END_VAR�	  (***FSM - TESTING and DISTRIBUTION***)
IF DistTest_Handler.ImmediateStop THEN
	Distribution_Handler.ImmediateStop:=TRUE;
	Testing_Handler.ImmediateStop:=TRUE;
ELSE
	Distribution_Handler.ImmediateStop:=FALSE;
	Testing_Handler.ImmediateStop:=FALSE;

	CASE state_DisTest OF
	
	DisTest_ready_to_initialize:
	   IF DistTest_Handler.Initialize THEN
	       Distribution_Handler.Initialize	:= TRUE;
		Testing_Handler.Initialize 		:= TRUE;
	
	       state_DisTest := DisTest_Initializing;
	   END_IF;
	
	DisTest_Initializing:
	   IF NOT Distribution_Handler.Initialize AND NOT Testing_Handler.Initialize THEN
	       DistTest_Handler.Initialize := FALSE;
	
	       state_DisTest := DisTest_ready_to_run;
	   END_IF;
	
	DisTest_ready_to_run:
	   IF DistTest_Handler.Run THEN
	       Distribution_Handler.Enable := TRUE;
	
	       state_DisTest := Distribution_enabling;
	   END_IF;
	
	Distribution_enabling:
	   IF NOT Distribution_Handler.Enable THEN
		Memory_Data := Save_data(Distribution_Index, Memory_Data, Distribution_Data);
	       Distribution_Handler.Disable := TRUE;
	
	       state_DisTest := Distribution_disabling;
	   END_IF;
	
	Distribution_disabling:
	   IF NOT Distribution_Handler.Disable THEN
		Memory_Data := Shift_data(Distribution_Index, Memory_Data);
		Testing_Data := Memory_Data[Testing_index];
	       Testing_Handler.Enable := TRUE;
	
	       state_DisTest := Testing_enabling;
	   END_IF;
	
	Testing_enabling:
		IF NOT Testing_Handler.Enable THEN
			Memory_Data := Save_data(Testing_index, Memory_Data, Testing_Data);
			Memory_Data[Testing_index].Discard:= Testing_colour(Memory_Data[Testing_index].Colour, Memory_Data[Testing_index].Height);
	
			IF  NOT Memory_Data[Testing_index].Discard THEN
				Testing_ready_to_send := TRUE;
	
				state_DisTest := Testing_waiting;
	
			ELSE
				Memory_Data[Testing_index].ID := 0;
				Testing_Data := Memory_Data[Testing_index];
				Testing_Handler.Disable := TRUE;
	
				state_DisTest := Testing_disabling;
			END_IF;
		END_IF;
	
	Testing_waiting:
	   IF Processing_ready_to_receive THEN
		Testing_Data := Memory_Data[Testing_index];
		Testing_ready_to_send 	:= FALSE;
		Testing_Handler.Disable 	:= TRUE;
	
	       state_DisTest := Testing_disabling;
	   END_IF;
	
	Testing_disabling:
	   IF NOT Testing_Handler.Disable THEN
		Memory_Data := Shift_data(Testing_index, Memory_Data);
		Processing_ready_to_receive:=FALSE;
	
	       state_DisTest := DisTest_ready_to_run;
	   END_IF;
	
	END_CASE;
END_IF

               N   , ��           DrillingUnit_PRG ��d	��d      ��������          PROGRAM DrillingUnit_PRG
(* Da ELIMINARE*)
VAR
	OperationType : INT := INIT;
END_VAR

VAR
   state: DrillingUnit_States := Drilling_ready_to_initialize;
END_VAR

(* Handler - Between PROCESSING_Prg and INSPECTION_UNIT_Prg*)
VAR_EXTERNAL
	Drilling_Handler : Subsystem_Handler;
END_VAR

(* GDs - Instances and Handler request *)
VAR
   Holding : Generic_Device;
   Holding_enable_request : BOOL;
   Holding_disable_request : BOOL;
   Holding_not_initialized : BOOL;

   Drill_Machine : Generic_Device;
   Drill_Machine_enable_request : BOOL;
   Drill_Machine_disable_request : BOOL;
   Drill_Machine_not_initialized : BOOL;

   Drilling : Generic_Device;
   Drilling_enable_request : BOOL;
   Drilling_disable_request : BOOL;
   Drilling_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
   (* Drill_Machine - DA_DF *)
     enable_Drill_Machine : BOOL;
     disable_Drill_Machine : BOOL;
     Drill_Machine_enabled : BOOL;
     Drill_Machine_disabled : BOOL;
     Drill_Machine_EnabledSensorFault : BOOL;
     Drill_Machine_DisabledSensorFault : BOOL;
     Drill_Machine_fault :BOOL;
     Drill_Machine_ActuatorFault : BOOL;

   (* Drilling - SA_NF *)
     enable_Drilling : BOOL;
     Drilling_ActuatorFault : BOOL;

  (* Holding - SA_NF *)
     enable_Holding : BOOL;
     Holding_ActuatorFault : BOOL;

   (* Pure sensors *)
   AvailableLoadInDrillingPositioning_Logical : BOOL;
   DrillingUnitDown_Logical : BOOL;
   DrillingUnitUp_Logical : BOOL;
END_VAR  IF NOT Drilling_Handler.ImmediateStop THEN

	CASE state OF
	
	Drilling_ready_to_initialize:
	   IF Drilling_Handler.Initialize THEN
		OperationType := INIT;
	
		state := Drilling_initializing;
	   END_IF;
	
	Drilling_initializing:
	   IF (NOT Holding_not_initialized AND NOT Drill_Machine_not_initialized AND NOT Drilling_not_initialized) THEN
		OperationType := RUN;
		Drilling_Handler.Initialize := FALSE;
	
		state := Drilling_ready_to_enable;
	   END_IF;
	
	Drilling_ready_to_enable:
	   IF Drilling_Handler.Enable AND AvailableLoadInDrillingPositioning_Logical THEN
	       Holding_enable_request := TRUE;
	
	       state := Holding_of_workpiece_enabling;
	   END_IF;
	
	Holding_of_workpiece_enabling:
	   IF NOT Holding_enable_request THEN
	       Drilling_enable_request := TRUE;
	
	       state := Drilling_enabling;
	   END_IF;
	
	Drilling_enabling:
	   IF NOT Drilling_enable_request THEN
	       Drill_Machine_enable_request := TRUE; (*Gli dico di andare gi�*)
	
	       state := Drill_Machine_moving_down;
	   END_IF;
	
	Drill_Machine_moving_down:
	   IF NOT Drill_Machine_enable_request THEN
		Drill_Machine_disable_request := TRUE; (* Gli dico di tornare su*)
	
	       state := Drill_Machine_moving_up;
	   END_IF;
	
	Drill_Machine_moving_up:
	   IF NOT Drill_Machine_disable_request THEN
	       Drilling_disable_request := TRUE;
	
	       state := Drilling_disabling;
	   END_IF;
	
	Drilling_disabling:
	   IF NOT Drilling_disable_request THEN
	       Drilling_Handler.Enable := FALSE;
	
	       state := Drilling_ready_to_disable;
	   END_IF;
	
	Drilling_ready_to_disable:
	   IF Drilling_Handler.Disable THEN
	       Holding_disable_request := TRUE;
	
	       state := Holding_of_workpiece_disabling;
	   END_IF;
	
	Holding_of_workpiece_disabling:
	   IF NOT Holding_disable_request THEN
	       Drilling_Handler.Disable := FALSE;
	
	       state := Drilling_ready_to_enable;
	   END_IF;
	
	END_CASE;
END_IF


(*** GENERIC DEVICES ***)

Drill_Machine.DeviceOperation := OperationType;
Drill_Machine.DeviceClock := TRUE;
Drill_Machine.DeviceDiagnosticsEnabled := TRUE;
Drill_Machine.DeviceEnablePreset := FALSE;
Drill_Machine.DeviceEnabledSensor := Drill_Machine_enabled;
Drill_Machine.DeviceDisabledSensor := Drill_Machine_disabled;
Drill_Machine.DeviceEnableTime := Drill_Machine_EnableTime;
Drill_Machine.DeviceDisableTime := Drill_Machine_DisableTime;
Drill_Machine.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_DOUBLE_ACTUATION;
Drill_Machine(DeviceEnableRequest := Drill_Machine_enable_request, DeviceDisableRequest := Drill_Machine_disable_request );
enable_Drill_Machine := Drill_Machine.EnableDevice;
Drill_Machine_not_initialized:=Drill_Machine.DeviceNotInitialized;
disable_Drill_Machine := Drill_Machine.DisableDevice;
Drill_Machine_ActuatorFault := Drill_Machine.DeviceActuatorFault;
Drill_Machine_EnabledSensorFault := Drill_Machine.DeviceEnabledSensorFault;
Drill_Machine_DisabledSensorFault := Drill_Machine.DeviceDisabledSensorFault;
Drill_Machine_fault := Drill_Machine.DeviceFault;


Drilling.DeviceOperation := OperationType;
Drilling.DeviceClock := TRUE;
Drilling.DeviceDiagnosticsEnabled := TRUE;
Drilling.DeviceEnablePreset := FALSE;
Drilling.DeviceEnableTime := Drilling_EnableTime;
Drilling.DeviceDisableTime := Drilling_DisableTime;
Drilling.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
Drilling(DeviceEnableRequest := Drilling_enable_request, DeviceDisableRequest := Drilling_disable_request );
enable_Drilling := Drilling.EnableDevice;
Drilling_not_initialized:=Drilling.DeviceNotInitialized;
Drilling_ActuatorFault := Drilling.DeviceActuatorFault;

Holding.DeviceOperation := OperationType;
Holding.DeviceClock := TRUE;
Holding.DeviceDiagnosticsEnabled := TRUE;
Holding.DeviceEnablePreset := FALSE;
Holding.DeviceEnableTime := Holding_EnableTime;
Holding.DeviceDisableTime := Holding_DisableTime;
Holding.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
Holding(DeviceEnableRequest := Holding_enable_request, DeviceDisableRequest := Holding_disable_request );
enable_Holding := Holding.EnableDevice;
Holding_not_initialized:=Holding.DeviceNotInitialized;
Holding_ActuatorFault := Holding.DeviceActuatorFault;
               Q   ,   8=           Generic_Device ��d	��d      ��������          FUNCTION_BLOCK Generic_Device
(* Follow the Generic_Device.lib of slide 45 *)
VAR_INPUT
	DeviceOperation : INT;
	DeviceType : BYTE;
	DeviceEnabledSensor : BOOL;
	DeviceDisabledSensor : BOOL;
	DeviceClock : BOOL;
	DeviceEnableTime : INT;
	DeviceDisableTime : INT;
	DeviceDiagnosticsEnabled : BOOL;
	DeviceEnablePreset : BOOL;
END_VAR


VAR_IN_OUT
	DeviceEnableRequest : BOOL;
	DeviceDisableRequest : BOOL;
END_VAR

VAR_OUTPUT
	EnableDevice : BOOL;
	DisableDevice : BOOL;
	DeviceEnabledSensorFault : BOOL;
	DeviceDisabledSensorFault : BOOL;
	DeviceActuatorFault : BOOL;
	DeviceFault : BOOL;
	DeviceNotInitialized : BOOL;
END_VAR

VAR_EXTERNAL
	INIT : INT:=0;
	RUN : INT:=1;
END_VAR

VAR_EXTERNAL
	(*Feedback*)
	DEVICE_WITHOUT_FEEDBACK : BYTE := 2#01000000;
	DEVICE_WITH_ENABLE_FEEDBACK : BYTE := 2#00010000;
	DEVICE_WITH_DISABLE_FEEDBACK : BYTE := 2#00100000;
	DEVICE_WITH_DOUBLE_FEEDBACK : BYTE := 2#00110000;
	DEVICE_FEEDBACK_MASK : BYTE := 2#11110000;
	(*Actuation*)
	DEVICE_WITH_SINGLE_ACTUATION : BYTE := 2#00000001;
	DEVICE_WITH_DOUBLE_ACTUATION : BYTE := 2#00000011;
	DEVICE_WITH_DA_NO_RETAIN : BYTE := 2#00000010;
	DEVICE_ACTUATION_MASK : BYTE := 2#00001111;
END_VAR

VAR
	(*DeviceState : INT;*)
	DeviceState : GenericDevice_States;
	DeviceTimer : INT;
	DeviceTimeout : BOOL;
	temp: BYTE;
	temp_int: INT;

	(*TImeout: BOOL; (* new variable*)*)
END_VAR

(* Nuove variabili per compensare differenze nel codice del prof *)
VAR
	DeviceDisabled : BOOL;
	DeviceEnabled : BOOL;
END_VAR
c  (* Nuove variabili per compensare differenze nel codice del prof *)
DeviceDisabled := DeviceDisabledSensor;
DeviceEnabled := DeviceEnabledSensor;


(*** DEVICE INITIALIZATION HANDLER, SLIDE 43 and 44 ***)

IF (DeviceOperation = INIT) THEN
	DeviceEnableRequest := FALSE;
	DeviceDisableRequest := FALSE;
	IF DeviceEnablePreset THEN
		EnableDevice := TRUE;
		DeviceTimer := DeviceEnableTime;
		IF (NOT (DeviceDisabled AND DeviceEnabled)) THEN
				DeviceState := DeviceEnablePhaseState;
		ELSE
				DeviceState := DeviceEnabledState;
		END_IF;
		DeviceNotInitialized := NOT DeviceEnabled OR DeviceDisabled;
	ELSE
		EnableDevice := FALSE;
		DeviceTimer := DeviceDisableTime;
		IF (NOT (DeviceDisabled AND DeviceEnabled)) THEN
			DeviceState := DeviceDisablePhaseState;
		ELSE
			DeviceState := DeviceDisabledState;
		END_IF;
		DeviceNotInitialized := DeviceEnabled OR NOT DeviceDisabled;
	END_IF;

	CASE (DeviceType AND DEVICE_FEEDBACK_MASK) OF
		DEVICE_WITH_ENABLE_FEEDBACK:
			DeviceDisabled := NOT DeviceEnabled;
		DEVICE_WITH_DISABLE_FEEDBACK :
			DeviceEnabled := NOT DeviceDisabled;
		DEVICE_WITHOUT_FEEDBACK:
			DeviceEnabled := DeviceEnablePreset;
			DeviceDisabled := NOT DeviceEnabled;
	END_CASE

END_IF;

(*** DEVICE CLOCK HANDLER, SLIDE 8 ***)
IF (DeviceClock AND (DeviceTimer > 0)) THEN
	DeviceTimer := DeviceTimer - 1;
END_IF;

(*** DEVICE SENSORS HANDLER, SLIDE 34 ***)
(** USED SLIDE 39**)
CASE (DeviceType AND DEVICE_FEEDBACK_MASK) OF
	DEVICE_WITH_DISABLE_FEEDBACK:
		 DeviceEnabledSensor := NOT DeviceDisabledSensor AND ((DeviceState = DeviceEnabledState) OR (DeviceState = DeviceEnablePhaseState) AND DeviceTimeout);
 	DEVICE_WITH_ENABLE_FEEDBACK:
		DeviceDisabledSensor := NOT DeviceEnabledSensor AND ((DeviceState = DeviceDisabledState) OR (DeviceState = DeviceDisablePhaseState) AND DeviceTimeout);
 	DEVICE_WITHOUT_FEEDBACK:
		DeviceEnabledSensor := (DeviceState = DeviceEnabledState) OR ((DeviceState = DeviceEnablePhaseState) AND DeviceTimeout);
		DeviceDisabledSensor := (DeviceState = DeviceDisabledState) OR ((DeviceState = DeviceDisablePhaseState) AND DeviceTimeout);
END_CASE;

(*** DEVICE CONTROL HANDLER, SLIDE 8 and 9 ***)
CASE (DeviceState) OF

	DeviceDisabledState:
		DeviceDisableRequest := FALSE;
		IF (DeviceEnableRequest) THEN
			EnableDevice := TRUE;
			DeviceTimer := DeviceEnableTime;
			DeviceState := DeviceEnablePhaseState;
		END_IF;

	DeviceEnablePhaseState:
		IF (DeviceDisableRequest) THEN
			DeviceEnableRequest := FALSE;
			EnableDevice := FALSE;
			DeviceTimer := DeviceDisableTime;
			DeviceState := DeviceDisablePhaseState;
		ELSIF (DeviceEnabled) THEN
			DeviceEnableRequest := FALSE;
			DeviceTimer := 0;
			DeviceState := DeviceEnabledState;
		END_IF;

	DeviceEnabledState:
		DeviceEnableRequest := FALSE;
		IF (DeviceDisableRequest) THEN
			EnableDevice := FALSE;
			DeviceTimer := DeviceDisableTime;
			DeviceState := DeviceDisablePhaseState;
		END_IF;

	DeviceDisablePhaseState:
		IF (DeviceEnableRequest) THEN
			DeviceDisableRequest := FALSE;
			EnableDevice := TRUE;
			DeviceTimer := DeviceEnableTime;
			DeviceState := DeviceEnablePhaseState;
		ELSIF (DeviceDisabled) THEN
			DeviceDisableRequest := FALSE;
			DeviceTimer := 0;
			DeviceState := DeviceDisabledState;
		END_IF;

END_CASE;

(*** DEVICE DIAGNOSTICS HANDLER, SLIDE 26 ***)
DeviceTimeout := (DeviceTimer = 0);

CASE (DeviceType AND DEVICE_FEEDBACK_MASK) OF
	DEVICE_WITH_DISABLE_FEEDBACK:
		 DeviceDisabledSensorFault := (NOT EnableDevice AND NOT DeviceDisabled AND NOT DeviceEnabled OR  EnableDevice AND DeviceDisabled AND DeviceEnabled) AND DeviceTimeout;
 	DEVICE_WITH_ENABLE_FEEDBACK:
		DeviceEnabledSensorFault := (NOT EnableDevice AND DeviceDisabled AND DeviceEnabled OR EnableDevice AND NOT DeviceDisabled AND NOT DeviceEnabled) AND DeviceTimeout;
 	DEVICE_WITH_DOUBLE_FEEDBACK:
		DeviceDisabledSensorFault := (NOT EnableDevice AND NOT DeviceDisabled AND NOT DeviceEnabled OR  EnableDevice AND DeviceDisabled AND DeviceEnabled) AND DeviceTimeout;
		DeviceEnabledSensorFault := (NOT EnableDevice AND DeviceDisabled AND DeviceEnabled OR EnableDevice AND NOT DeviceDisabled AND NOT DeviceEnabled) AND DeviceTimeout;
END_CASE;

DeviceActuatorFault := (NOT EnableDevice AND NOT DeviceDisabled AND DeviceEnabled OR EnableDevice AND DeviceDisabled AND NOT DeviceEnabled) AND DeviceTimeout;

DeviceFault := DeviceDisabledSensorFault OR DeviceEnabledSensorFault OR DeviceActuatorFault;

(*** DEVICE ACTUATORS HANDLER, SLIDE 38 ***)
CASE (DeviceType AND DEVICE_ACTUATION_MASK) OF
	(*DEVICE_WITH_SINGLE_ACTUATION: nothing to do, just use EnableDevice*)
	DEVICE_WITH_DOUBLE_ACTUATION:
		DisableDevice := NOT EnableDevice;
	DEVICE_WITH_DA_NO_RETAIN:
		EnableDevice := EnableDevice AND (DeviceState = DeviceEnablePhaseState);
		DisableDevice := NOT EnableDevice AND (DeviceState = DeviceDisablePhaseState);
END_CASE;               I   , �� )�           InspectionUnit_PRG ��d	��d      ��������        �  PROGRAM InspectionUnit_PRG
(* Da ELIMINARE*)
VAR
	OperationType : INT := INIT;
END_VAR


VAR
	state: InspectionUnit_States := Inspection_ready_to_initialize;
END_VAR

(* Handler - Between PROCESSING_Prg and INSPECTION_UNIT_Prg*)
VAR_EXTERNAL
	Inspection_Handler : Subsystem_Handler;
	Inspection_SubData : Data_Handler;
END_VAR

(* GDs - Instances and Handler request *)
VAR
   CylinderToInspect : Generic_Device;
   CylinderToInspect_enable_request : BOOL;
   CylinderToInspect_disable_request : BOOL;
   CylinderToInspect_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(* CylinderToInspect - SA_NF *)
     enable_CylinderToInspect : BOOL;
     CylinderToInspect_ActuatorFault : BOOL;

	(* Pure Sensors *)
	AvailableLoadInControlPositionLogical : BOOL := FALSE;
	InControlLoadInWrongPositionToBeDrilledLogical : BOOL := FALSE; (* True se orientamento giusto*)
END_VAR�  IF NOT Inspection_Handler.ImmediateStop THEN

	CASE state OF
	
	Inspection_ready_to_initialize:
	   IF Inspection_Handler.Initialize THEN
		OperationType := INIT;
	
		state := Inspection_initializing;
	   END_IF;
	
	Inspection_initializing:
	   IF (NOT CylinderToInspect_not_initialized) THEN
		OperationType := RUN;
		Inspection_Handler.Initialize := FALSE;
	
		state := Inspection_ready_to_enable;
	   END_IF;
	
	Inspection_ready_to_enable:
	   IF Inspection_Handler.Enable AND AvailableLoadInControlPositionLogical THEN
	       CylinderToInspect_enable_request := TRUE;
	
	       state := Cylinder_enabling;
	   END_IF;
	
	Cylinder_enabling:
	IF NOT CylinderToInspect_enable_request THEN
		IF InControlLoadInWrongPositionToBeDrilledLogical THEN
			Inspection_SubData.Orientation := TRUE; (* Correctly oriented, NOT to discard*)
		END_IF;
		IF NOT InControlLoadInWrongPositionToBeDrilledLogical THEN
			Inspection_SubData.Orientation := FALSE;
		END_IF;
		Inspection_Handler.Enable := FALSE;
	
	       state := Inspection_ready_to_disable;
	END_IF;
	
	Inspection_ready_to_disable:
	   IF Inspection_Handler.Disable THEN
	       CylinderToInspect_disable_request := TRUE;
	
	       state := Cylinder_disabling;
	   END_IF;
	
	Cylinder_disabling:
	   IF NOT CylinderToInspect_disable_request THEN
	       Inspection_Handler.Disable := FALSE;
	
	       state := Inspection_ready_to_enable;
	   END_IF;
	END_CASE;
END_IF


(*** GENERIC DEVICES ***)
CylinderToInspect.DeviceOperation := OperationType;
CylinderToInspect.DeviceClock := TRUE;
CylinderToInspect.DeviceDiagnosticsEnabled := TRUE;
CylinderToInspect.DeviceEnablePreset := FALSE;
CylinderToInspect.DeviceEnableTime := CylinderToInspect_EnableTime;
CylinderToInspect.DeviceDisableTime := CylinderToInspect_DisableTime;
CylinderToInspect.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
CylinderToInspect(DeviceEnableRequest := CylinderToInspect_enable_request, DeviceDisableRequest := CylinderToInspect_disable_request );
enable_CylinderToInspect := CylinderToInspect.EnableDevice;
CylinderToInspect_not_initialized:=CylinderToInspect.DeviceNotInitialized;
CylinderToInspect_ActuatorFault := CylinderToInspect.DeviceActuatorFault;
               a   ,  '�           ItemsAssembling_PRG ��d	��d      ��������        c  PROGRAM ItemsAssembling_PRG
VAR
	OperationType : INT := INIT;
	state : ItemsAssemblingUnit_States := ItemsAssemblingUnit_ready_to_initialize;
END_VAR

(*Between ASSEMBLY_Prg and ITEMASSEMBLING_Prg*)
VAR_EXTERNAL
	ItemsAssembling_Handler : Subsystem_Handler;
END_VAR


(* GDs - Instances and  Handler request*)
VAR
   R_Initial_position : Generic_Device; 	(*R1*)
   R_Initial_position_enable_request : BOOL;
   R_Initial_position_disable_request : BOOL;
   R_Initial_position_not_initialized : BOOL;

   R_TakeSpring : Generic_Device; (*R6*)
   R_TakeSpring_enable_request : BOOL;
   R_TakeSpring_disable_request : BOOL;
   R_TakeSpring_not_initialized : BOOL;

   R_TakePiston : Generic_Device; (*R7*)
   R_TakePiston_enable_request : BOOL;
   R_TakePiston_disable_request : BOOL;
   R_TakePiston_not_initialized : BOOL;

   R_TakeCover : Generic_Device; (*R8*)
   R_TakeCover_enable_request : BOOL;
   R_TakeCover_disable_request : BOOL;
   R_TakeCover_not_initialized : BOOL;

   R_AssemblyPosition : Generic_Device; (*R9*)
   R_AssemblyPosition_enable_request : BOOL;
   R_AssemblyPosition_disable_request : BOOL;
   R_AssemblyPosition_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
   (* GDs- Robot - SA-SAF*)
	(*R1*)
   enable_R_Initial_position_A : BOOL;
   R_Initial_position_enabled_A : BOOL;
   R_Initial_position_fault_A :BOOL;
	(*R6*)
   enable_R_TakeSpring : BOOL;
   R_TakeSpring_enabled : BOOL;
   R_TakeSpring_fault :BOOL;

	(*R7*)
   enable_R_TakePiston : BOOL;
   R_TakePiston_enabled : BOOL;
   R_TakePiston_fault :BOOL;

	(*R8*)
   enable_R_TakeCover : BOOL;
   R_TakeCover_enabled : BOOL;
   R_TakeCover_fault :BOOL;

	(*R9*)
   enable_R_AssemblyPosition : BOOL;
   R_AssemblyPosition_enabled : BOOL;
   R_AssemblyPosition_fault :BOOL;
END_VAR


�  IF NOT ItemsAssembling_Handler.ImmediateStop THEN

	CASE state OF
	ItemsAssemblingUnit_ready_to_initialize:
		IF ItemsAssembling_Handler.Initialize THEN
			OperationType := INIT;
	
			state := ItemsAssemblingUnit_initializing;
		END_IF;
	
	ItemsAssemblingUnit_initializing:
		IF (NOT R_TakeSpring_not_initialized AND NOT R_TakePiston_not_initialized AND NOT R_TakeCover_not_initialized AND NOT R_AssemblyPosition_not_initialized AND NOT R_Initial_position_not_initialized) THEN
			OperationType := RUN;
			ItemsAssembling_Handler.Initialize := FALSE;
	
			state := ItemsAssemblingUnit_ready_to_enable;
		END_IF;
	
	ItemsAssemblingUnit_ready_to_enable:
		IF ItemsAssembling_Handler.Enable THEN
			R_TakePiston_enable_request := TRUE;
	
			state := TakePiston_enabling;
		END_IF;
	
	TakePiston_enabling:
		IF NOT R_TakePiston_enable_request THEN
			R_TakePiston_disable_request := TRUE;
	
			state := TakePiston_disabling;
		END_IF;
	
	TakePiston_disabling:
		IF NOT R_TakePiston_disable_request THEN
			R_AssemblyPosition_enable_request := TRUE;
	
			state := MoveToAssembly_enabling_with_piston;
		END_IF;
	
	MoveToAssembly_enabling_with_piston:
		IF NOT R_AssemblyPosition_enable_request THEN
			R_AssemblyPosition_disable_request := TRUE;
	
			state := MoveToAssembly_disabling_with_piston;
		END_IF;
	
	MoveToAssembly_disabling_with_piston:
		IF NOT R_AssemblyPosition_disable_request THEN
			R_TakeSpring_enable_request := TRUE;
	
			state := TakeSpring_enabling;
		END_IF;
	
	TakeSpring_enabling:
		IF NOT R_TakeSpring_enable_request THEN
			R_TakeSpring_disable_request := TRUE;
	
			state := TakeSpring_disabling;
		END_IF;
	
	TakeSpring_disabling:
		IF NOT R_TakeSpring_disable_request THEN
			R_AssemblyPosition_enable_request := TRUE;
	
			state := MoveToAssembly_enabling_with_spring;
		END_IF;
	
	MoveToAssembly_enabling_with_spring:
		IF NOT R_AssemblyPosition_enable_request THEN
			R_AssemblyPosition_disable_request := TRUE;
	
			state := MoveToAssembly_disabling_with_spring;
		END_IF;
	
	MoveToAssembly_disabling_with_spring:
		IF NOT R_AssemblyPosition_disable_request THEN
			R_TakeCover_enable_request := TRUE;
	
			state := TakeCover_enabling;
		END_IF;
	
	TakeCover_enabling:
		IF NOT R_TakeCover_enable_request THEN
			R_TakeCover_disable_request := TRUE;
	
			state := TakeCover_disabling;
		END_IF;
	
	TakeCover_disabling:
		IF NOT R_TakeCover_disable_request THEN
			R_AssemblyPosition_enable_request := TRUE;
	
			state := MoveToAssembly_enabling_with_cover;
		END_IF;
	
	MoveToAssembly_enabling_with_cover:
		IF NOT R_AssemblyPosition_enable_request THEN
			R_AssemblyPosition_disable_request := TRUE;
	
			state := MoveToAssembly_disabling_with_cover;
		END_IF;
	
	MoveToAssembly_disabling_with_cover:
		IF NOT R_AssemblyPosition_disable_request THEN
			ItemsAssembling_Handler.Enable := FALSE;
	
			state := ItemsAssemblingUnit_ready_to_disable;
		END_IF;
	
	ItemsAssemblingUnit_ready_to_disable:
		IF ItemsAssembling_Handler.Disable THEN
			R_Initial_position_enable_request := TRUE;
	
			state := Take_finished_piece_enabling;
		END_IF;
	
	Take_finished_piece_enabling:
		IF NOT R_Initial_position_enable_request THEN
			R_Initial_position_disable_request := TRUE;
	
			state := Take_finished_piece_disabling;
		END_IF;
	
	Take_finished_piece_disabling:
		IF NOT R_Initial_position_disable_request THEN
			ItemsAssembling_Handler.Disable := FALSE;
	
			state := ItemsAssemblingUnit_ready_to_enable;
		END_IF;
	
	END_CASE;
END_IF


(* GENERIC DEVICES *)
R_TakeSpring.DeviceOperation := OperationType;
R_TakeSpring.DeviceClock := TRUE;
R_TakeSpring.DeviceDiagnosticsEnabled := TRUE;
R_TakeSpring.DeviceEnablePreset := FALSE;
R_TakeSpring.DeviceEnabledSensor := R_TakeSpring_enabled;
R_TakeSpring.DeviceEnableTime := 110;
R_TakeSpring.DeviceDisableTime := 3;
R_TakeSpring.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_TakeSpring(DeviceEnableRequest := R_TakeSpring_enable_request, DeviceDisableRequest := R_TakeSpring_disable_request );
enable_R_TakeSpring := R_TakeSpring.EnableDevice;
R_TakeSpring_not_initialized:=R_TakeSpring.DeviceNotInitialized;
R_TakeSpring_fault := R_TakeSpring.DeviceFault;

R_TakePiston.DeviceOperation := OperationType;
R_TakePiston.DeviceClock := TRUE;
R_TakePiston.DeviceDiagnosticsEnabled := TRUE;
R_TakePiston.DeviceEnablePreset := FALSE;
R_TakePiston.DeviceEnabledSensor := R_TakePiston_enabled;
R_TakePiston.DeviceEnableTime := 110;
R_TakePiston.DeviceDisableTime := 3;
R_TakePiston.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_TakePiston(DeviceEnableRequest := R_TakePiston_enable_request, DeviceDisableRequest := R_TakePiston_disable_request );
enable_R_TakePiston := R_TakePiston.EnableDevice;
R_TakePiston_not_initialized:=R_TakePiston.DeviceNotInitialized;
R_TakePiston_fault := R_TakePiston.DeviceFault;

R_TakeCover.DeviceOperation := OperationType;
R_TakeCover.DeviceClock := TRUE;
R_TakeCover.DeviceDiagnosticsEnabled := TRUE;
R_TakeCover.DeviceEnablePreset := FALSE;
R_TakeCover.DeviceEnabledSensor := R_TakeCover_enabled;
R_TakeCover.DeviceEnableTime := 110;
R_TakeCover.DeviceDisableTime := 3;
R_TakeCover.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_TakeCover(DeviceEnableRequest := R_TakeCover_enable_request, DeviceDisableRequest := R_TakeCover_disable_request );
enable_R_TakeCover := R_TakeCover.EnableDevice;
R_TakeCover_not_initialized:=R_TakeCover.DeviceNotInitialized;
R_TakeCover_fault := R_TakeCover.DeviceFault;

R_AssemblyPosition.DeviceOperation := OperationType;
R_AssemblyPosition.DeviceClock := TRUE;
R_AssemblyPosition.DeviceDiagnosticsEnabled := TRUE;
R_AssemblyPosition.DeviceEnablePreset := FALSE;
R_AssemblyPosition.DeviceEnabledSensor := R_AssemblyPosition_enabled;
R_AssemblyPosition.DeviceEnableTime := 110;
R_AssemblyPosition.DeviceDisableTime := 3;
R_AssemblyPosition.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_AssemblyPosition(DeviceEnableRequest := R_AssemblyPosition_enable_request, DeviceDisableRequest := R_AssemblyPosition_disable_request );
enable_R_AssemblyPosition := R_AssemblyPosition.EnableDevice;
R_AssemblyPosition_not_initialized:=R_AssemblyPosition.DeviceNotInitialized;
R_AssemblyPosition_fault := R_AssemblyPosition.DeviceFault;

R_Initial_position.DeviceOperation := OperationType;
R_Initial_position.DeviceClock := TRUE;
R_Initial_position.DeviceDiagnosticsEnabled := TRUE;
R_Initial_position.DeviceEnablePreset := FALSE;
R_Initial_position.DeviceEnabledSensor := R_Initial_position_enabled_A;
R_Initial_position.DeviceEnableTime := 110;
R_Initial_position.DeviceDisableTime := 10;
R_Initial_position.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Initial_position(DeviceEnableRequest := R_Initial_position_enable_request, DeviceDisableRequest := R_Initial_position_disable_request );
enable_R_Initial_position_A := R_Initial_position.EnableDevice;
R_Initial_position_not_initialized:=R_Initial_position.DeviceNotInitialized;
R_Initial_position_fault_A := R_Initial_position.DeviceFault;
               m   , �R �6           ItemsSupply_PRG ��d	��d      ��������        L  PROGRAM ItemsSupply_PRG
VAR
	OperationType : INT := INIT;
	state : ItemsSupplyUnit_States := ItemsSupplyUnit_ready_to_initialize;
END_VAR

(* Between ASSEMBLY_Prg and ITEMSSUPPLY_Prg*)
VAR_EXTERNAL
	ItemsSupply_Handler : 		Subsystem_Handler;
	ItemsSupply_SubData:		Data_Handler;
END_VAR


(* GDs - Instances and  Handler request*)
VAR
	PistonSelector : Generic_Device;
	PistonSelector_enable_request : BOOL;
	PistonSelector_disable_request : BOOL;
	PistonSelector_not_initialized : BOOL;

	ExtractCover : Generic_Device;
	ExtractCover_enable_request : BOOL;
	ExtractCover_disable_request : BOOL;
	ExtractCover_not_initialized : BOOL;

	ExtractSpring : Generic_Device;
	ExtractSpring_enable_request : BOOL;
	ExtractSpring_disable_request : BOOL;
	ExtractSpring_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(*PistonSelector - DA-DF*)
     enable_PistonSelector : BOOL;
     disable_PistonSelector : BOOL;
     PistonSelector_enabled : BOOL;
     PistonSelector_disabled : BOOL;
     PistonSelector_EnabledSensorFault : BOOL;
     PistonSelector_DisabledSensorFault : BOOL;
     PistonSelector_fault :BOOL;
     PistonSelector_ActuatorFault : BOOL;

	(*ExtractCover - SA_DF*)
     enable_ExtractCover : BOOL;
     ExtractCover_enabled : BOOL;
     ExtractCover_disabled : BOOL;
     ExtractCover_EnabledSensorFault : BOOL;
     ExtractCover_DisabledSensorFault : BOOL;
     ExtractCover_fault :BOOL;
     ExtractCover_ActuatorFault : BOOL;

	(*ExtractSpring - SA-DF*)
     enable_ExtractSpring : BOOL;
     ExtractSpring_enabled : BOOL;
     ExtractSpring_disabled : BOOL;
     ExtractSpring_EnabledSensorFault : BOOL;
     ExtractSpring_DisabledSensorFault : BOOL;
     ExtractSpring_fault :BOOL;
     ExtractSpring_ActuatorFault : BOOL;
END_VAR�  IF NOT ItemsSupply_Handler.ImmediateStop THEN

	CASE state OF
	ItemsSupplyUnit_ready_to_initialize:
		IF ItemsSupply_Handler.Initialize THEN
			OperationType := INIT;
	
			state := ItemsSupplyUnit_initializing;
		END_IF;
	
	ItemsSupplyUnit_initializing:
		IF (NOT ExtractCover_not_initialized AND NOT ExtractSpring_not_initialized) THEN
			OperationType := RUN;
			ItemsSupply_Handler.Initialize := FALSE;
	
			state := ItemsSupplyUnit_ready_to_enable;
		END_IF;
	
	ItemsSupplyUnit_ready_to_enable:
		IF ItemsSupply_Handler.Enable THEN
			ExtractCover_enable_request := TRUE;
			ExtractSpring_enable_request := TRUE;
			PistonSelector_enable_request := TRUE; (*Pistone Silver *)
	
			state := ItemsSupplyUnit_enabling;
		END_IF;
	
	ItemsSupplyUnit_enabling:
		IF (NOT ExtractCover_enable_request AND NOT ExtractSpring_enable_request AND NOT PistonSelector_enable_request) THEN
			IF NOT ItemsSupply_SubData.Colour THEN (* Black piece -> needs Silver piston*)
				ItemsSupply_Handler.Enable := FALSE;
	
				state := ItemsSupplyUnit_ready_to_disable;
			ELSE  (* Red/Silver piece -> needs Black piston*)
				PistonSelector_disable_request := TRUE;
	
				state := PistonSelector_disabling_for_RedSilver_piece;
			END_IF;
		END_IF
	
	PistonSelector_disabling_for_RedSilver_piece:
		IF NOT PistonSelector_disable_request THEN (*Pistone Black *)
				ItemsSupply_Handler.Enable := FALSE;
	
				state := ItemsSupplyUnit_ready_to_disable;
		END_IF
	
	ItemsSupplyUnit_ready_to_disable:
		IF ItemsSupply_Handler.Disable THEN
			ExtractCover_disable_request := TRUE;
			ExtractSpring_disable_request := TRUE;
			PistonSelector_disable_request := TRUE; (*Disable utile nel caso di Black piece, senn� ridondandte*)
	
			state := ItemsSupplyUnit_disabling;
		END_IF;
	
	ItemsSupplyUnit_disabling:
		IF (NOT ExtractCover_disable_request AND NOT ExtractSpring_disable_request AND NOT PistonSelector_disable_request) THEN
			ItemsSupply_Handler.Disable := FALSE;
	
			state := ItemsSupplyUnit_ready_to_enable;
		END_IF;
	
	END_CASE;
END_IF

(*** GENERIC DEVICES ***)
PistonSelector.DeviceOperation := OperationType;
PistonSelector.DeviceClock := TRUE;
PistonSelector.DeviceDiagnosticsEnabled := TRUE;
PistonSelector.DeviceEnablePreset := FALSE;
PistonSelector.DeviceEnabledSensor := PistonSelector_enabled;
PistonSelector.DeviceDisabledSensor := PistonSelector_disabled;
PistonSelector.DeviceEnableTime := PistonSelector_EnableTime;
PistonSelector.DeviceDisableTime := PistonSelector_DisableTime;
PistonSelector.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_DOUBLE_ACTUATION;
PistonSelector(DeviceEnableRequest := PistonSelector_enable_request, DeviceDisableRequest := PistonSelector_disable_request );
enable_PistonSelector := PistonSelector.EnableDevice;
PistonSelector_not_initialized:=PistonSelector.DeviceNotInitialized;
disable_PistonSelector := PistonSelector.DisableDevice;
PistonSelector_ActuatorFault := PistonSelector.DeviceActuatorFault;
PistonSelector_EnabledSensorFault := PistonSelector.DeviceEnabledSensorFault;
PistonSelector_DisabledSensorFault := PistonSelector.DeviceDisabledSensorFault;
PistonSelector_fault := PistonSelector.DeviceFault;

ExtractCover.DeviceOperation := OperationType;
ExtractCover.DeviceClock := TRUE;
ExtractCover.DeviceDiagnosticsEnabled := TRUE;
ExtractCover.DeviceEnablePreset := FALSE;
ExtractCover.DeviceEnabledSensor := ExtractCover_enabled;
ExtractCover.DeviceDisabledSensor := ExtractCover_disabled;
ExtractCover.DeviceEnableTime := ExtractCover_EnableTime;
ExtractCover.DeviceDisableTime := ExtractCover_DisableTime;
ExtractCover.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
ExtractCover(DeviceEnableRequest := ExtractCover_enable_request, DeviceDisableRequest := ExtractCover_disable_request );
enable_ExtractCover := ExtractCover.EnableDevice;
ExtractCover_not_initialized:=ExtractCover.DeviceNotInitialized;
ExtractCover_ActuatorFault := ExtractCover.DeviceActuatorFault;
ExtractCover_EnabledSensorFault := ExtractCover.DeviceEnabledSensorFault;
ExtractCover_DisabledSensorFault := ExtractCover.DeviceDisabledSensorFault;
ExtractCover_fault := ExtractCover.DeviceFault;

ExtractSpring.DeviceOperation := OperationType;
ExtractSpring.DeviceClock := TRUE;
ExtractSpring.DeviceDiagnosticsEnabled := TRUE;
ExtractSpring.DeviceEnablePreset := FALSE;
ExtractSpring.DeviceEnabledSensor := ExtractSpring_enabled;
ExtractSpring.DeviceDisabledSensor := ExtractSpring_disabled;
ExtractSpring.DeviceEnableTime := ExtractSpring_EnableTime;
ExtractSpring.DeviceDisableTime := ExtractSpring_DisableTime;
ExtractSpring.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
ExtractSpring(DeviceEnableRequest := ExtractSpring_enable_request, DeviceDisableRequest := ExtractSpring_disable_request );
enable_ExtractSpring := ExtractSpring.EnableDevice;
ExtractSpring_not_initialized:=ExtractSpring.DeviceNotInitialized;
ExtractSpring_ActuatorFault := ExtractSpring.DeviceActuatorFault;
ExtractSpring_EnabledSensorFault := ExtractSpring.DeviceEnabledSensorFault;
ExtractSpring_DisabledSensorFault := ExtractSpring.DeviceDisabledSensorFault;
ExtractSpring_fault := ExtractSpring.DeviceFault;
               %   , � >@{           Lights ��d	��d      ��������        �  PROGRAM Lights
VAR
END_VAR

(*logical variables*)
VAR_EXTERNAL
	ResetSignalsEnable: BOOL;

	LightEmptyWarehouseLogical: BOOL;
	LightEmptyCoverhouseLogical: BOOL; (*in assembly station*)

	LightStartLogical: BOOL;
END_VAR

(*physical variables*)
VAR_EXTERNAL
	LightStart:BOOL;
	LightReset:BOOL;
	LightEmptyWarehouse: BOOL;
	LightEmptyCoverhouse: BOOL; (*in assembly station*)

END_VAR�   LightReset:=ResetSignalsEnable;

LightEmptyWarehouse:= LightEmptyWarehouseLogical;

LightEmptyCoverhouse:= LightEmptyCoverhouseLogical;

LightStart:=LightStartLogical;

               \   ,  � �           Main_PRG ��d	��d      ��������        �  PROGRAM Main_PRG
VAR
	state: Main_States := Ready_to_initialize;
END_VAR

(*Output - Logical*)
VAR_EXTERNAL
	Init_Logical : BOOL;
	Start_Logical : BOOL;
	Reset_Logical : BOOL;
	Stop_Logical : BOOL;
	OnPhaseStop_Logical: BOOL;
	ImmediateStop_Logical:BOOL;
END_VAR


(* Between MAIN_Prg and DISTTEST_Prg *)
VAR_EXTERNAL
	DistTest_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Making_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and ROBOT_Prg *)
VAR_EXTERNAL
	Robot_Handler : 	System_Handler;
END_VAR

(* Between MAIN_Prg and SIGNALMANAGEMENT_Prg *)
VAR_EXTERNAL
	SignalManagement_Handler : 	System_Handler;
END_VAR

(*Light*)
VAR_EXTERNAL
	LightStartLogical: BOOL;
END_VAR

N  (*FSM*)
CASE state OF

Ready_to_initialize:
	IF Init_Logical THEN
		DistTest_Handler.Initialize 	:= TRUE;
		Making_Handler.Initialize 		:= TRUE;
		Robot_Handler.Initialize 		:= TRUE;
		SignalManagement_Handler.Initialize := TRUE;

		state := Initializing;
	END_IF

Initializing:
	IF NOT DistTest_Handler.Initialize AND NOT Making_Handler.Initialize AND NOT Robot_Handler.Initialize THEN

		state := Ready_to_Run;
	END_IF

Ready_to_Run:
	IF Start_Logical THEN
		SignalManagement_Handler.Initialize := FALSE; (* Perch� on abbiamo un suo akn, tanto ci mette un colpo di clock *)
		SignalManagement_Handler.Run	:= TRUE;
		DistTest_Handler.Run 				:= TRUE;
		Making_Handler.Run 				:= TRUE;
		Robot_Handler.Run					:= TRUE;

		state := Running;
	END_IF

Running:
	IF ImmediateStop_Logical THEN
		DistTest_Handler.ImmediateStop 	:= TRUE;
		Making_Handler.ImmediateStop	:= TRUE;
		Robot_Handler.ImmediateStop		:= TRUE;

		state := ImmediateStopping;

	ELSIF OnPhaseStop_Logical THEN
		DistTest_Handler.Run 	:= FALSE;
		Making_Handler.Run 	:= FALSE;
		Robot_Handler.Run		:= FALSE;

		state := OnPhaseStopping;

	ELSIF Stop_Logical THEN
		DistTest_Handler.Run 	:= FALSE;

		state := Stopping;
	END_IF

ImmediateStopping:
	IF (Start_Logical AND NOT ImmediateStop_Logical) THEN
		DistTest_Handler.ImmediateStop 	:= FALSE;
		Making_Handler.ImmediateStop	:= FALSE;
		Robot_Handler.ImmediateStop		:= FALSE;

		state := Ready_to_Run;
	END_IF

OnPhaseStopping:
	IF (Start_Logical AND NOT OnPhaseStop_Logical) THEN
		DistTest_Handler.Run 	:= TRUE;
		Making_Handler.Run 	:= TRUE;
		Robot_Handler.Run		:= TRUE;


		state := Ready_to_Run;
	END_IF


Stopping:
	IF (Start_Logical AND NOT Stop_Logical) THEN
		DistTest_Handler.Run 	:= TRUE;
		Making_Handler.Run 	:= TRUE;
		Robot_Handler.Run		:= TRUE;


		state := Ready_to_Run;
	END_IF

END_CASE;

IF (state=Ready_to_Run) OR ((state=ImmediateStopping) AND NOT ImmediateStop_Logical) OR ((state=OnPhaseStopping) AND NOT OnPhaseStop_Logical) THEN
	LightStartLogical:=TRUE;
ELSE
	LightStartLogical:=FALSE;
END_IF
               L   ,  �J�        
   Making_PRG ��d	��d      ��������        �  PROGRAM Making_PRG
(* Memory array to manage data memory and its operation *)
VAR_EXTERNAL
    	Memory_Data: ARRAY [1..8] OF Data_Handler;
END_VAR

(* Arrey's index *)
VAR_EXTERNAL CONSTANT
	Distribution_index:		UINT:=1;
	Testing_index:			UINT:=2;
	Rotary_index:			UINT:=3;
	Inspection_index:			UINT:=4;
	Drilling_index:			UINT:=5;
	Expelling_index:			UINT:=6;
	PickandPlace_index:		UINT:=7;
	Supply_index:			UINT:=8;
END_VAR



(* Between MAIN_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Making_Handler : 	System_Handler;
END_VAR

(* Between MAKING_Prg and PROCESSING_Prg *)
VAR_EXTERNAL
	Processing_Handler:		Subsystem_Handler;
	Rotary_Data:			Data_Handler;
	Inspection_Data:			Data_Handler;
	Drilling_Data:			Data_Handler;
	Expelling_Data:			Data_Handler;
END_VAR

(* Between DISTTEST_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Testing_ready_to_send :BOOL:= FALSE;
END_VAR

(* Between MAKING_Prg and DISTTEST_Prg *)
VAR_EXTERNAL
      Processing_ready_to_receive :BOOL:= FALSE;
END_VAR

(* Between ASSEMBLY_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Robot_ready_to_receive:BOOL:= TRUE;
END_VAR





VAR
   state_Making: Making_States := M_Processing_ready_to_initialize;
END_VAR�  (**FSM - MAKING**)
IF Making_Handler.ImmediateStop THEN
	Processing_Handler.ImmediateStop:=TRUE;
ELSE
	Processing_Handler.ImmediateStop:=FALSE;

	CASE state_Making OF
	
	M_Processing_ready_to_initialize:
	   IF Making_Handler.Initialize THEN
	       Processing_Handler.Initialize := TRUE;
	
	       state_Making := M_Processing_Initializing;
	   END_IF;
	
	M_Processing_Initializing:
	   IF NOT Processing_Handler.Initialize THEN
	       Making_Handler.Initialize := FALSE;
	
	       state_Making := M_Processing_ready_to_run;
	   END_IF;
	
	M_Processing_ready_to_run: (* IF Run AND (There is an item to process) *)
	  IF Making_Handler.Run AND
		(Testing_ready_to_send OR INT_TO_BOOL(Memory_Data[Rotary_index].ID) OR INT_TO_BOOL(Memory_Data[Inspection_index].ID) OR INT_TO_BOOL(Memory_Data[Drilling_index].ID) OR INT_TO_BOOL(Memory_Data[Expelling_index].ID) )
	  THEN
			IF Testing_ready_to_send THEN
		      	Processing_ready_to_receive := TRUE;
	
		       	state_Making := Processing_waiting_to_receive;
			ELSE
				Rotary_Data := 		Memory_Data[Rotary_index];
				Inspection_Data:= 	Memory_Data[Inspection_index];
				Drilling_Data := 		Memory_Data[Drilling_index];
				Expelling_Data := 	Memory_Data[Expelling_index];
				Processing_Handler.Enable := TRUE;
		
				state_Making := Processing_enabling;
			END_IF;
	   END_IF;
	
	Processing_waiting_to_receive:
	   IF NOT Processing_ready_to_receive THEN
		Rotary_Data := 		Memory_Data[Rotary_index];
		Inspection_Data:= 	Memory_Data[Inspection_index];
		Drilling_Data := 		Memory_Data[Drilling_index];
		Expelling_Data := 	Memory_Data[Expelling_index];
	      Processing_Handler.Enable := TRUE;
	
	      state_Making := Processing_enabling;
	   END_IF;
	
	Processing_enabling:
	   IF NOT Processing_Handler.Enable THEN
		Memory_Data := Save_data(Rotary_index, Memory_Data, Rotary_Data);
		Memory_Data := Save_data(Inspection_index, Memory_Data, Inspection_Data);
		Memory_Data := Save_data(Drilling_index, Memory_Data, Drilling_Data);
		Memory_Data := Save_data(Expelling_index, Memory_Data, Expelling_Data);
		IF INT_TO_BOOL(Memory_Data[Inspection_index].ID) THEN
			Memory_Data[Inspection_index].Discard:= Testing_orientation(Memory_Data[Inspection_index].Orientation);
		END_IF;
	
		IF INT_TO_BOOL(Memory_Data[Expelling_index].ID) THEN
		       state_Making := Processing_waiting_to_expel;
		ELSE
			Processing_Handler.Disable := TRUE;
	
			state_Making := Processing_disabling;
		END_IF;
	   END_IF;
	
	Processing_waiting_to_expel:
	   IF Robot_ready_to_receive THEN
		Robot_ready_to_receive:= FALSE;
		Processing_Handler.Disable := TRUE;
	
	       state_Making := Processing_disabling;
	   END_IF;
	
	Processing_disabling:
	   IF NOT Processing_Handler.Disable THEN
		IF INT_TO_BOOL(Memory_Data[Expelling_index].ID) THEN
			Memory_Data := Shift_data(Expelling_index, Memory_Data);
		END_IF
	
	       state_Making := M_Processing_ready_to_run;
	   END_IF;
	
	END_CASE;
END_IF
               (   , � � g           PickandPlace_PRG ��d	��d      ��������        �  PROGRAM PickandPlace_PRG

VAR
	OperationType : INT := INIT;
END_VAR

VAR
   state: PickandPlace_States := PickandPlace_ready_to_initialize;
END_VAR

(* Handler - Between MACHINE_Prg and PICKANDPLACE_Prg*)
VAR_EXTERNAL
	PickandPlace_Handler : Subsystem_Handler;
	PickandPlace_Data : Data_Handler;
END_VAR

VAR
	(*R1*)
   R_Initial_position : Generic_Device;
   R_Initial_position_enable_request : BOOL;
   R_Initial_position_disable_request : BOOL;
   R_Initial_position_not_initialized : BOOL;

	(*R2*)
   R_Take_black_piece : Generic_Device;
   R_Take_black_piece_enable_request : BOOL;
   R_Take_black_piece_disable_request : BOOL;
   R_Take_black_piece_not_initialized : BOOL;

	(*R3*)
   R_Take_redsilver_piece : Generic_Device;
   R_Take_redsilver_piece_enable_request : BOOL;
   R_Take_redsilver_piece_disable_request : BOOL;
   R_Take_redsilver_piece_not_initialized : BOOL;

	(*R4*)
   R_Take_black_upsidedown_piece : Generic_Device;
   R_Take_black_upsidedown_piece_enable_request : BOOL;
   R_Take_black_upsidedown_piece_disable_request : BOOL;
   R_Take_black_upsidedown_piece_not_initialized : BOOL;

	(*R5*)
   R_Take_redsilver_upsidedown_piece : Generic_Device;
   R_Take_redsilver_upsidedown_piece_enable_request : BOOL;
   R_Take_redsilver_upsidedown_piece_disable_request : BOOL;
   R_Take_redsilver_upsidedown_piece_not_initialized : BOOL;
END_VAR


VAR_EXTERNAL
(* GDs- Robot - SA-SAF*)
	(*R1*)
   enable_R_Initial_position : BOOL;
   R_Initial_position_enabled : BOOL;
   R_Initial_position_fault :BOOL;

	(*R2*)
   enable_R_Take_black_piece : BOOL;
   R_Take_black_piece_enabled : BOOL;
   R_Take_black_piece_fault :BOOL;

	(*R3*)
   enable_R_Take_redsilver_piece : BOOL;
   R_Take_redsilver_piece_enabled : BOOL;
   R_Take_redsilver_piece_fault :BOOL;

	(*R4*)
   enable_R_Take_black_upsidedown_piece : BOOL;
   R_Take_black_upsidedown_piece_enabled : BOOL;
   R_Take_black_upsidedown_piece_fault :BOOL;

	(*R5*)
   enable_R_Take_redsilver_upsidedown_piece : BOOL;
   R_Take_redsilver_upsidedown_piece_enabled : BOOL;
   R_Take_redsilver_upsidedown_piece_fault :BOOL;

	(* Pure sensors*)
	AvailableLoadForRobot_Logical_PP : BOOL;
END_VARc!  IF NOT PickandPlace_Handler.ImmediateStop THEN

	CASE state OF
	
	PickandPlace_ready_to_initialize:
		IF PickandPlace_Handler.Initialize THEN
			OperationType := INIT;
	
			state := PickandPlace_initializing;
		END_IF;
	
	PickandPlace_initializing:
		IF (NOT R_Initial_position_not_initialized AND NOT R_Take_black_piece_not_initialized AND NOT R_Take_redsilver_piece_not_initialized AND NOT R_Take_black_upsidedown_piece_not_initialized AND NOT R_Take_redsilver_upsidedown_piece_not_initialized) THEN
			OperationType := RUN;
			PickandPlace_Handler.Initialize := FALSE;
	
			state := PickandPlace_ready_to_enable;
		END_IF;
	
	PickandPlace_ready_to_enable:
		IF PickandPlace_Handler.Enable AND AvailableLoadForRobot_Logical_PP THEN
			R_Initial_position_enable_request := TRUE;
	
			state := Robot_to_initial_position_enabling;
		END_IF;
	
	Robot_to_initial_position_enabling:
		IF NOT R_Initial_position_enable_request THEN
			R_Initial_position_disable_request := TRUE;
	
			state := Robot_to_initial_position_disabling;
		END_IF;
	
	Robot_to_initial_position_disabling:
		IF NOT R_Initial_position_disable_request THEN
			PickandPlace_Handler.Enable := FALSE;
	
			state := PickandPlace_ready_to_disable;
		END_IF;
	
	PickandPlace_ready_to_disable:
		IF PickandPlace_Handler.Disable THEN
	
			IF PickandPlace_Data.Discard THEN (* Wrong oriented - to discard *)
				IF PickandPlace_Data.Colour THEN (* Red/Metallic*)
					R_Take_redsilver_upsidedown_piece_enable_request := TRUE;
	
					state := Discarding_wrong_oriented_redsilver_enabling;
	
				ELSE (* Black *)
					R_Take_black_upsidedown_piece_enable_request := TRUE;
	
					state := Discarding_wrong_oriented_black_enabling;
				END_IF;
	
			ELSE; (* Correctly oriented *)
				IF PickandPlace_Data.Colour THEN (* Red/Metallic*)
					R_Take_redsilver_piece_enable_request := TRUE;
	
					state := Taking_correctly_oriented_redsilver_enabling;
	
				ELSE (* Black *)
					R_Take_black_piece_enable_request := TRUE;
	
					state := Taking_correctly_oriented_black_enabling;
				END_IF;
			END_IF;
		END_IF;
	
	
	Discarding_wrong_oriented_redsilver_enabling:
		IF NOT R_Take_redsilver_upsidedown_piece_enable_request THEN
			R_Take_redsilver_upsidedown_piece_disable_request := TRUE;
	
			state := Discarding_wrong_oriented_redsilver_disabling;
		END_IF;
	
	Discarding_wrong_oriented_redsilver_disabling:
		IF NOT R_Take_redsilver_upsidedown_piece_disable_request THEN
			PickandPlace_Handler.Disable := FALSE;
	
			state := PickandPlace_ready_to_enable;
		END_IF;
	
	
	Discarding_wrong_oriented_black_enabling:
		IF NOT R_Take_black_upsidedown_piece_enable_request THEN
			R_Take_black_upsidedown_piece_disable_request := TRUE;
	
			state := Discarding_wrong_oriented_black_disabling;
		END_IF;
	
	Discarding_wrong_oriented_black_disabling:
		IF NOT R_Take_black_upsidedown_piece_disable_request THEN
			PickandPlace_Handler.Disable := FALSE;
	
			state := PickandPlace_ready_to_enable;
		END_IF;
	
	Taking_correctly_oriented_redsilver_enabling:
		IF NOT R_Take_redsilver_piece_enable_request THEN
			R_Take_redsilver_piece_disable_request := TRUE;
	
			state := Taking_correctly_oriented_redsilver_disabling;
		END_IF;
	
	Taking_correctly_oriented_redsilver_disabling:
		IF NOT R_Take_redsilver_piece_disable_request THEN
			PickandPlace_Handler.Disable := FALSE;
	
			state := PickandPlace_ready_to_enable;
		END_IF;
	
	
	Taking_correctly_oriented_black_enabling:
		IF NOT R_Take_black_piece_enable_request THEN
			R_Take_black_piece_disable_request := TRUE;
	
			state := Taking_correctly_oriented_black_disabling;
		END_IF;
	
	Taking_correctly_oriented_black_disabling:
		IF NOT R_Take_black_piece_disable_request THEN
			PickandPlace_Handler.Disable := FALSE;
	
			state := PickandPlace_ready_to_enable;
		END_IF;
	
	END_CASE;
END_IF

(*** GENERIC DEVICES ***)

R_Initial_position.DeviceOperation := OperationType;
R_Initial_position.DeviceClock := TRUE;
R_Initial_position.DeviceDiagnosticsEnabled := TRUE;
R_Initial_position.DeviceEnablePreset := FALSE;
R_Initial_position.DeviceEnabledSensor := R_Initial_position_enabled;
R_Initial_position.DeviceEnableTime := 110;
R_Initial_position.DeviceDisableTime := 10;
R_Initial_position.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Initial_position(DeviceEnableRequest := R_Initial_position_enable_request, DeviceDisableRequest := R_Initial_position_disable_request );
enable_R_Initial_position := R_Initial_position.EnableDevice;
R_Initial_position_not_initialized:=R_Initial_position.DeviceNotInitialized;
R_Initial_position_fault := R_Initial_position.DeviceFault;

R_Take_black_piece.DeviceOperation := OperationType;
R_Take_black_piece.DeviceClock := TRUE;
R_Take_black_piece.DeviceDiagnosticsEnabled := TRUE;
R_Take_black_piece.DeviceEnablePreset := FALSE;
R_Take_black_piece.DeviceEnabledSensor := R_Take_black_piece_enabled;
R_Take_black_piece.DeviceEnableTime := 110;
R_Take_black_piece.DeviceDisableTime := 10;
R_Take_black_piece.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Take_black_piece(DeviceEnableRequest := R_Take_black_piece_enable_request, DeviceDisableRequest := R_Take_black_piece_disable_request );
enable_R_Take_black_piece := R_Take_black_piece.EnableDevice;
R_Take_black_piece_not_initialized:=R_Take_black_piece.DeviceNotInitialized;
R_Take_black_piece_fault := R_Take_black_piece.DeviceFault;

R_Take_redsilver_piece.DeviceOperation := OperationType;
R_Take_redsilver_piece.DeviceClock := TRUE;
R_Take_redsilver_piece.DeviceDiagnosticsEnabled := TRUE;
R_Take_redsilver_piece.DeviceEnablePreset := FALSE;
R_Take_redsilver_piece.DeviceEnabledSensor := R_Take_redsilver_piece_enabled;
R_Take_redsilver_piece.DeviceEnableTime := 110;
R_Take_redsilver_piece.DeviceDisableTime := 10;
R_Take_redsilver_piece.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Take_redsilver_piece(DeviceEnableRequest := R_Take_redsilver_piece_enable_request, DeviceDisableRequest := R_Take_redsilver_piece_disable_request );
enable_R_Take_redsilver_piece := R_Take_redsilver_piece.EnableDevice;
R_Take_redsilver_piece_not_initialized:=R_Take_redsilver_piece.DeviceNotInitialized;
R_Take_redsilver_piece_fault := R_Take_redsilver_piece.DeviceFault;

R_Take_black_upsidedown_piece.DeviceOperation := OperationType;
R_Take_black_upsidedown_piece.DeviceClock := TRUE;
R_Take_black_upsidedown_piece.DeviceDiagnosticsEnabled := TRUE;
R_Take_black_upsidedown_piece.DeviceEnablePreset := FALSE;
R_Take_black_upsidedown_piece.DeviceEnabledSensor := R_Take_black_upsidedown_piece_enabled;
R_Take_black_upsidedown_piece.DeviceEnableTime := 150;
R_Take_black_upsidedown_piece.DeviceDisableTime := 50;
R_Take_black_upsidedown_piece.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Take_black_upsidedown_piece(DeviceEnableRequest := R_Take_black_upsidedown_piece_enable_request, DeviceDisableRequest := R_Take_black_upsidedown_piece_disable_request );
enable_R_Take_black_upsidedown_piece := R_Take_black_upsidedown_piece.EnableDevice;
R_Take_black_upsidedown_piece_not_initialized:=R_Take_black_upsidedown_piece.DeviceNotInitialized;
R_Take_black_upsidedown_piece_fault := R_Take_black_upsidedown_piece.DeviceFault;

R_Take_redsilver_upsidedown_piece.DeviceOperation := OperationType;
R_Take_redsilver_upsidedown_piece.DeviceClock := TRUE;
R_Take_redsilver_upsidedown_piece.DeviceDiagnosticsEnabled := TRUE;
R_Take_redsilver_upsidedown_piece.DeviceEnablePreset := FALSE;
R_Take_redsilver_upsidedown_piece.DeviceEnabledSensor := R_Take_redsilver_upsidedown_piece_enabled;
R_Take_redsilver_upsidedown_piece.DeviceEnableTime := 150;
R_Take_redsilver_upsidedown_piece.DeviceDisableTime := 50;
R_Take_redsilver_upsidedown_piece.DeviceType := DEVICE_WITH_ENABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
R_Take_redsilver_upsidedown_piece(DeviceEnableRequest := R_Take_redsilver_upsidedown_piece_enable_request, DeviceDisableRequest := R_Take_redsilver_upsidedown_piece_disable_request );
enable_R_Take_redsilver_upsidedown_piece := R_Take_redsilver_upsidedown_piece.EnableDevice;
R_Take_redsilver_upsidedown_piece_not_initialized:=R_Take_redsilver_upsidedown_piece.DeviceNotInitialized;
R_Take_redsilver_upsidedown_piece_fault := R_Take_redsilver_upsidedown_piece.DeviceFault;               �  ,  t �X           PlantAssemblaggio ��d	��d      ��������        �  PROGRAM PlantAssemblaggio
VAR
	State: INT;
	Count: BOOL;
	Count2: BOOL;
	CaseBlack: INT;
	CaseRedSilver: INT;
	CaseOverturned: INT;
	InitialPosition: INT;
	FLAGToExtractSpringInAssemblyStationBlockHigh: BOOL;
	FLAGBlockingCylinderForwardInAssemblyStationBlockHigh: BOOL;
	FLAGToExtractCoverInAssemblyStationForwardBlockHigh: BOOL;
	Count3: BOOL;
	Count4: BOOL;
END_VAR
L�  (*QUANDO UN PEZZO E' STATO ASSEMBLATO, RESETTO TUTTI I CONTATORI*)
IF ElementAssembled
THEN	CanDiscard:=FALSE; (*RENDE INVISIBILE IL CESTO PORTA SCARTI*)
		ElementToDiscard:=FALSE; (*RESET DELLE PROPRIETA' DEL PEZZO NEL CESTO DEGLI SCARTI*)
		ElementToDiscardRed:=FALSE;
		ElementToDiscardBlack:=FALSE;
		ElementToDiscardSilver:=FALSE;
		ElementToDiscardPosition:=0;
		CoverToDiscard:=FALSE;
		PistonBlackToDiscard:=FALSE;
		PistonSilverToDiscard:=FALSE;
		ElementAssembled:=FALSE;
END_IF;

(*Premendo il tasto FillWarehouse, si simula il riempimento di tutti i magazzini*)
IF FillAllWarehouses
THEN 	SpringLoad:=TRUE;
		CoverLoad:=TRUE;
		PistonLoad:=TRUE;
		FillAllWarehouses:=FALSE;
END_IF;


(*MODULO MOLLE*)

(*Tasto che serve a caricare le molle nel magazzino*)
IF SpringLoad
THEN	SpringNumber:=8;
		Spring1:=TRUE;
		Spring2:=TRUE;
		Spring3:=TRUE;
		Spring4:=TRUE;
		Spring5:=TRUE;
		Spring6:=TRUE;
		Spring7:=TRUE;
		Spring8:=TRUE;
		SpringLoad:=FALSE;
END_IF;

(*NB: IL CILINDRO DI ESTRAZIONE E' NORMALMENTE APERTO!!!*)
(*Quando il comando di estrazione � TRUE, lo stelo del cilindro rientra*)
IF FLAGToExtractSpringInAssemblyStationBlockHigh OR (ToExtractSpringInAssemblyStation AND NOT ToExtractSpringInAssemblyStationBlockLow)
THEN	ExtractSpringPosition:=ExtractSpringPosition-4;
		IF ToExtractSpringInAssemblyStationBlockHigh
		THEN FLAGToExtractSpringInAssemblyStationBlockHigh:=TRUE;
		ELSE FLAGToExtractSpringInAssemblyStationBlockHigh:=FALSE;
		END_IF;
ELSE	ExtractSpringPosition:=ExtractSpringPosition+4;
END_IF;

(*Quando il pistone di estrazione molle e' arrivato a fine corsa, viene resa visibile la molla estratta*)
IF ExtractSpringPosition<=0
THEN	ExtractSpringPosition:=0;

		IF NOT ToExtractSpringInAssemblyStationInRetroactivePositionBlockLow
		THEN ToExtractSpringInAssemblyStationInRetroactivePosition:=TRUE;
		ELSE ToExtractSpringInAssemblyStationInRetroactivePosition:=FALSE;
		END_IF;

		IF SpringNumber>0 (*se il magazzino molle non � vuoto*)
		THEN VisualSpring:=TRUE; (*allora rendo visibile la molla che trasla insieme al pistone di prelievo molle*)
		END_IF;

ELSE	IF NOT ToExtractSpringInAssemblyStationInRetroactivePositionBlockHigh
		THEN ToExtractSpringInAssemblyStationInRetroactivePosition:=FALSE;
		ELSE ToExtractSpringInAssemblyStationInRetroactivePosition:=TRUE;
		END_IF;

IF ToExtractSpringInAssemblyStationInRetroactivePositionBlockHigh
THEN ToExtractSpringInAssemblyStationInRetroactivePosition:=TRUE;
END_IF;

END_IF;

(*Quando il pistone di estrazione molle e' in posizione di riposo, si attiva il relativo sensore e si disattiva il flag "SpringBehind", che indica il movimento verso il riposo del pistone*)
IF ExtractSpringPosition>=40
THEN	ExtractSpringPosition:=40;

		IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockLow
		THEN ToExtractSpringInAssemblyStationInExtensivePosition:=TRUE;
		ELSE ToExtractSpringInAssemblyStationInExtensivePosition:=FALSE;
		END_IF;

		VisualSpring:=FALSE;
		IF Count
		THEN 	SpringNumber:=SpringNumber-1;
				SpringExtract:=TRUE;
				Count:=FALSE;
		END_IF;

ELSE	IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh
		THEN ToExtractSpringInAssemblyStationInExtensivePosition:=FALSE;
		ELSE ToExtractSpringInAssemblyStationInExtensivePosition:=TRUE;
		END_IF;
		Count:=TRUE;
END_IF;

IF ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh
THEN ToExtractSpringInAssemblyStationInExtensivePosition:=TRUE;
END_IF;

IF SpringNumber=7 (*AND ToExtractSpringInAssemblyStationInRetroactivePosition*)
THEN Spring8:=FALSE;
END_IF;
IF SpringNumber=6
THEN Spring7:=FALSE;
END_IF;
IF SpringNumber=5
THEN Spring6:=FALSE;
END_IF;
IF SpringNumber=4
THEN Spring5:=FALSE;
END_IF;
IF SpringNumber=3
THEN Spring4:=FALSE;
END_IF;
IF SpringNumber=2
THEN Spring3:=FALSE;
END_IF;
IF SpringNumber=1
THEN Spring2:=FALSE;
END_IF;
IF SpringNumber=0
THEN 	Spring1:=FALSE;
END_IF;


(*MODULO PISTONI*)

IF PistonLoad
THEN 	PistonBlackLoad:=TRUE;
		PistonSilverLoad:=TRUE;
		PistonLoad:=FALSE;
END_IF;

(*Quando attiviamo questa variabile, riempiamo il magazzino pistoni neri*)
IF PistonBlackLoad
THEN	PistonBlackNumber:=8;
		PistonBlack1:=TRUE;
		PistonBlack2:=TRUE;
		PistonBlack3:=TRUE;
		PistonBlack4:=TRUE;
		PistonBlack5:=TRUE;
		PistonBlack6:=TRUE;
		PistonBlack7:=TRUE;
		PistonBlack8:=TRUE;
		PistonBlackLoad:=FALSE;
END_IF;

(*Quando attiviamo questa variabile, riempiamo il magazzino pistoni grigi*)
IF PistonSilverLoad
THEN	PistonSilverNumber:=8;
		PistonSilver1:=TRUE;
		PistonSilver2:=TRUE;
		PistonSilver3:=TRUE;
		PistonSilver4:=TRUE;
		PistonSilver5:=TRUE;
		PistonSilver6:=TRUE;
		PistonSilver7:=TRUE;
		PistonSilver8:=TRUE;
		PistonSilverLoad:=FALSE;
END_IF;

(*DA COMMENTARE...*)
	IF PistonSelectorIsOnTheRight
	THEN	PistonBlack8:=FALSE;
	END_IF;

	IF PistonSelectorIsOnTheLeft
	THEN	PistonSilver8:=FALSE;
	END_IF;

(*La sottrazione del pistone viene fatta dall'end-effector*)
IF PistonSelectorPosition=20
THEN	IF PistonSilverNumber=7
		THEN	PistonSilver7:=FALSE;
		END_IF;

		IF PistonSilverNumber=6
		THEN	PistonSilver6:=FALSE;
		END_IF;

		IF PistonSilverNumber=5
		THEN	PistonSilver5:=FALSE;
		END_IF;

		IF PistonSilverNumber=4
		THEN	PistonSilver4:=FALSE;
		END_IF;

		IF PistonSilverNumber=3
		THEN	PistonSilver3:=FALSE;
		END_IF;

		IF PistonSilverNumber=2
		THEN	PistonSilver2:=FALSE;
		END_IF;

		IF PistonSilverNumber=1
		THEN	PistonSilver1:=FALSE;
		END_IF;
END_IF;

IF PistonSelectorPosition=-20
THEN	IF PistonBlackNumber=7
		THEN	PistonBlack7:=FALSE;
		END_IF;

		IF PistonBlackNumber=6
		THEN	PistonBlack6:=FALSE;
		END_IF;

		IF PistonBlackNumber=5
		THEN	PistonBlack5:=FALSE;
		END_IF;

		IF PistonBlackNumber=4
		THEN	PistonBlack4:=FALSE;
		END_IF;

		IF PistonBlackNumber=3
		THEN	PistonBlack3:=FALSE;
		END_IF;

		IF PistonBlackNumber=2
		THEN	PistonBlack2:=FALSE;
		END_IF;

		IF PistonBlackNumber=1
		THEN	PistonBlack1:=FALSE;
		END_IF;
END_IF;

(*Incremento di posizione verso sinistra*)
IF PistonSelectorGoOnTheLeftBlockHigh OR (PistonSelectorGoOnTheLeft AND NOT PistonSelectorGoOnTheLeftBlockLow)
THEN	PistonSelectorPosition:=PistonSelectorPosition+3;
END_IF;
(*Incremento di posizione verso destra*)
IF PistonSelectorGoOnTheRightBlockHigh OR (PistonSelectorGoOnTheRight AND NOT PistonSelectorGoOnTheRightBlockLow)
THEN	PistonSelectorPosition:=PistonSelectorPosition-3;
END_IF;

(*Quando l'incremento di posizione giunge a 20 siamo in corrsipondenza del fine corsa destro*)
IF PistonSelectorPosition<=-20
THEN PistonSelectorPosition:=-20;
		(*Se il sensore di posizione destro non � bloccato alto, allora il sensore � attivo*)
		IF NOT PistonSelectorIsOnTheRightBlockLow
		THEN PistonSelectorIsOnTheRight:=TRUE;
		ELSE PistonSelectorIsOnTheRight:=FALSE;
		END_IF;
		(*Se il magazzino pistoni grigi non � vuoto, allora visualizzo il  pistone all'interno del selettore*)
		IF PistonBlackNumber>0
		THEN PistonBlackExtract:=TRUE;
		END_IF;

ELSE 	IF NOT PistonSelectorIsOnTheRightBlockHigh
		THEN PistonSelectorIsOnTheRight:=FALSE;
		END_IF;
END_IF;

IF PistonSelectorIsOnTheRightBlockHigh
THEN PistonSelectorIsOnTheRight:=TRUE;
END_IF;

IF PistonSelectorPosition>=20
THEN PistonSelectorPosition:=20;

		IF NOT PistonSelectorIsOnTheLeftBlockLow
		THEN PistonSelectorIsOnTheLeft:=TRUE;
		ELSE PistonSelectorIsOnTheLeft:=FALSE;
		END_IF;

		IF PistonSilverNumber>0
		THEN PistonSilverExtract:=TRUE;
		END_IF;

ELSE	IF NOT PistonSelectorIsOnTheLeftBlockHigh
		THEN PistonSelectorIsOnTheLeft:=FALSE;
		END_IF;
END_IF;

IF PistonSelectorIsOnTheLeftBlockHigh
THEN PistonSelectorIsOnTheLeft:=TRUE;
END_IF;


(*MODULO COPERCHI*)

IF CoverLoad
THEN	CoverNumber:=8;
		IF NOT EmptyCoverHouseInAssemblyStationBlockHigh
		THEN EmptyCoverHouseInAssemblyStation:=FALSE;
		END_IF;
		Cover1:=TRUE;
		Cover2:=TRUE;
		Cover3:=TRUE;
		Cover4:=TRUE;
		Cover5:=TRUE;
		Cover6:=TRUE;
		Cover7:=TRUE;
		Cover8:=TRUE;
		CoverLoad:=FALSE;
END_IF;

IF FLAGToExtractCoverInAssemblyStationForwardBlockHigh OR (ToExtractCoverInAssemblyStationForward AND NOT ToExtractCoverInAssemblyStationForwardBlockLow)
THEN	ExtractCoverPosition:=ExtractCoverPosition+3;
		IF ToExtractCoverInAssemblyStationForwardBlockHigh
		THEN FLAGToExtractCoverInAssemblyStationForwardBlockHigh:=TRUE;
		ELSE FLAGToExtractCoverInAssemblyStationForwardBlockHigh:=FALSE;
		END_IF;
ELSE	ExtractCoverPosition:=ExtractCoverPosition-3;
END_IF;

IF ExtractCoverPosition<=0
THEN	ExtractCoverPosition:=0;
		IF NOT ToExtractCoverInAssemblyStationInRetroactivePositionBlockLow
		THEN ToExtractCoverInAssemblyStationInRetroactivePosition:=TRUE;
		ELSE ToExtractCoverInAssemblyStationInRetroactivePosition:=FALSE;
		END_IF;

		IF NOT EmptyCoverHouseInAssemblyStation
		THEN Cover1:=TRUE;
		END_IF;

ELSE	IF NOT ToExtractCoverInAssemblyStationInRetroactivePositionBlockHigh
		THEN ToExtractCoverInAssemblyStationInRetroactivePosition:=FALSE;
		ELSE ToExtractCoverInAssemblyStationInRetroactivePosition:=TRUE;
		END_IF;
END_IF;

IF ExtractCoverPosition>=65
THEN	ExtractCoverPosition:=65;
		Cover1:=FALSE;
		IF Count2
		THEN 	CoverNumber:=CoverNumber-1;
				CoverExtract:=TRUE;
				Count2:=FALSE;
		END_IF;

		IF NOT ToExtractCoverInAssemblyStationInExtensivePositionBlockLow
		THEN ToExtractCoverInAssemblyStationInExtensivePosition:=TRUE;
		ELSE ToExtractCoverInAssemblyStationInExtensivePosition:=FALSE;
		END_IF;

ELSE	IF NOT ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh
		THEN ToExtractCoverInAssemblyStationInExtensivePosition:=FALSE;
		ELSE ToExtractCoverInAssemblyStationInExtensivePosition:=TRUE;
		END_IF;
		Count2:=TRUE;
END_IF;

IF  ToExtractCoverInAssemblyStationInRetroactivePositionBlockHigh
THEN ToExtractCoverInAssemblyStationInRetroactivePosition:=TRUE;
END_IF;

IF ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh
THEN ToExtractCoverInAssemblyStationInExtensivePosition:=TRUE;
END_IF;


IF ExtractCoverPosition=0
THEN	IF CoverNumber=7
		THEN	Cover8:=FALSE;
		END_IF;
		IF CoverNumber=6
		THEN	Cover7:=FALSE;
		END_IF;
		IF CoverNumber=5
		THEN	Cover6:=FALSE;
		END_IF;
		IF CoverNumber=4
		THEN	Cover5:=FALSE;
		END_IF;
		IF CoverNumber=3
		THEN	Cover4:=FALSE;
		END_IF;
		IF CoverNumber=2
		THEN	Cover3:=FALSE;
		END_IF;
		IF CoverNumber=1
		THEN	Cover2:=FALSE;
		END_IF;
		IF CoverNumber=0
		THEN	Cover1:=FALSE;
				IF NOT EmptyCoverHouseInAssemblyStationBlockLow
				THEN EmptyCoverHouseInAssemblyStation:=TRUE;
				ELSE EmptyCoverHouseInAssemblyStation:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF EmptyCoverHouseInAssemblyStationBlockHigh
THEN EmptyCoverHouseInAssemblyStation:=TRUE;
END_IF;


(*SIMULAZIONE DEI MOVIMENTI DEL ROBOT *)
(*________________________________________________________________________________________________________*)
(*
(*Bloccaggio basso degli attuatori "robot"*)
IF RobotTakeLoadToDiascardBlockLow OR RobotTakeRedSilverLoadBlockLow OR RobotTakeBlackLoadBlockLow OR RobotGoToInitialPositionBlockLow OR RobotGoToSpringHouseBlockLow OR RobotGoToPistonHouseBlockLow OR RobotGoToCoverHouseBlockLow OR RobotTakeCurrentLoadToAssemblyBlockLow
THEN RobotBitIn3:=FALSE; RobotBitIn2:=FALSE; RobotBitIn1:=FALSE; RobotBitIn0:=FALSE;
END_IF;
*)
IF Bit4Output
THEN State:=0;
END_IF;

(*Spingendo il tasto StopProgramRunning, simulo lo spegnimento del programma CosiRop*)
IF StopProgramRunning
THEN RobotProgramRunning:=FALSE;
ELSE RobotProgramRunning:=TRUE;
END_IF;


IF RobotProgramRunning
THEN
		(*Se si riceve il comando di prelievo di una base capovolta(Base Nera 0101, Base Rossa-Argento 0100) *)
		IF RobotTakeLoadToDiascardBlockHigh OR ((NOT Bit3Output AND Bit2Output AND NOT Bit1Output) AND NOT RobotTakeLoadToDiascardBlockLow)
		THEN	State:=1;
		END_IF;

		(*Se si riceve il comando di prelievo di una Base Nera 0010 *)
		IF RobotTakeBlackLoadBlockHigh OR ((NOT Bit3Output AND NOT Bit2Output AND Bit1Output  AND NOT Bit0Output) AND NOT RobotTakeBlackLoadBlockLow)
		THEN	State:=2;
		END_IF;

		(*Se si riceve il comando di prelievo di una Base Rossa-Argento 0011 *)
		IF RobotTakeRedSilverLoadBlockHigh OR ((NOT Bit3Output AND NOT Bit2Output AND Bit1Output AND Bit0Output) AND NOT RobotTakeRedSilverLoadBlockLow)
		THEN	State:=3;
		END_IF;

		IF (NOT Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND State=0
		THEN State:=4;
		END_IF;
END_IF;


CASE State OF
(*Lo stato "0" rappresenta lo stato di Idle*)
0:	CaseOverturned:=20;
	CaseBlack:=20;
	CaseRedSilver:=20;
	InitialPosition:=20;

(*Lo stato "1" rappresenta il caso di comando di prelievo di una base Overturned*)
1:	CASE CaseOverturned OF
								(*Da posizione iniziale a stazione1 per prelevare la base*)
						20:		IF NOT RobotEngineBlockLow (*Se non ci sono guasti nei motori del robot, allora il robot comincia a muoversi*)
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10; (*SCENDI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF NOT RobotInInitialPositionBlockHigh (*Se non c'� un bloccaggio alto del sensoe virtuale di posizione iniziale, allora...*)
								THEN Bit3Input:=FALSE; Bit2Input:=FALSE; Bit1Input:=FALSE; Bit0Input:=FALSE; (*RobotInInitialPosition:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition>=170
								THEN	RobotGoVerticalPosition:=170;
										CaseOverturned:=21;
								END_IF;
								(*Presa della base*)
						21:		EndEffectorPosition:=EndEffectorPosition+1; (*CHIUDI END-EFFECTOR*)
								IF EndEffectorPosition>=3
								THEN 	EndEffectorPosition:=3;
										CaseOverturned:=22;
								END_IF;
								(*Buffer di trasferimento delle informazioni relative alla base*)
						22:		ElementInEndEffector:=TRUE; (*MOSTRA LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorBlack:=ElementStation1RobotBlack;
								ElementInEndEffectorSilver:=ElementStation1RobotSilver;
								ElementInEndEffectorRed:=ElementStation1RobotRed;
								ElementInEndEffectorOverturned:=ElementStation1RobotOverturned;
								IF ElementInEndEffectorOverturned
								THEN ElementInEndEffectorO:='O';
								ELSE ElementInEndEffectorO:='';
								END_IF;
								CaseOverturned:=23;

						23:		ElementStation1RobotCharged:=FALSE;
								CaseOverturned:=24;

						24:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10; (*SALI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN	RobotGoVerticalPosition:=0;
										CaseOverturned:=25;
								END_IF;
								(*RESETTO LE INFORMAZIONI ALLA STAZIONE1 PRIMA DI METTERE A FALSE AvailableLoadForRobot IN MODO DA AVERE LE GIUSTE INFORMAZIONI SUL PEZZO SUCCESSIVO*)
						25:		ElementStation1RobotRed:=FALSE;
								ElementStation1RobotBlack:=FALSE;
								ElementStation1RobotSilver:=FALSE;
								ElementStation1RobotOverturned:=FALSE;
								IF NOT AvailableLoadForRobotBlockHigh
								THEN	AvailableLoadForRobot:=FALSE; (*RESET DEL SENSORE DI PRESENZA NELLA STAZIONE1, A QUESTO PUNTO LA GIOSTRA PUO' MANDARE UN'ALTRO PEZZO*)
								END_IF;
								CaseOverturned:=26;

						26:		CanText:='Scarti'; (*Mostra il testo "scarti" all'interno del cesto*)
								CanColour:=FALSE; (*Quando CanColour � FALSE, il colore del cesto � rosso. Quando � TRUE il colore del cesto � verde*)
								CanDiscard:=TRUE; (*VARIABILE DI VISUALIZZAZIONE: rende visibile il cesto contenente gli scarti*)
								CaseOverturned:=27;
								(*Ritorno alla posizione iniziale*)
						27:		IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10; (*MUOVITI ORIZZONTALMENTE VERSO DESTRA*)
								END_IF;
								IF RobotGoHorizontalPosition>=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseOverturned:=28;
								END_IF;
								(*MOSTRA IL PEZZO CHE FINIRA' NELLA STAZIONE DI SCARTO*)
						28:		ElementToDiscard:=TRUE;
								IF ElementInEndEffectorOverturned
								THEN ElementToDiscardO:='O';
								END_IF;
								ElementToDiscardRed:=ElementInEndEffectorRed;
								ElementToDiscardBlack:=ElementInEndEffectorBlack;
								ElementToDiscardSilver:=ElementInEndEffectorSilver;
								CaseOverturned:=29;


						29:		EndEffectorPosition:=EndEffectorPosition-1; (*APRI END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN 	EndEffectorPosition:=0;
										CaseOverturned:=30;
								END_IF;
								(*RESET DELLE BASI NELL'END-EFFECTOR, NON L'HO FATTO NELL'IF PRECEDENTE PERCHE' AVREI RESO INVISIBILI LE BASI NELLA STAZIONE SCARTI*)
						30:		ElementInEndEffector:=FALSE; (*RENDO INVISIBILE LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorRed:=FALSE;
								ElementInEndEffectorBlack:=FALSE;
								ElementInEndEffectorSilver:=FALSE;
								ElementInEndEffectorOverturned:=FALSE;
								CaseOverturned:=31;

						31:		ElementToDiscardPosition:=ElementToDiscardPosition+5; (*lo scarto scende fino a finire nel cestino*)
								IF ElementToDiscardPosition>=80
								THEN 	ElementToDiscardPosition:=80;
										CaseOverturned:=32;
								END_IF;

						32:		IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-5; (*MUOVITI ORIZZONTALMENTE VERSO SINISTRA*)
								END_IF;
								IF RobotGoHorizontalPosition<=0
								THEN 	RobotGoHorizontalPosition:=0;
										CaseOverturned:=33;
								END_IF;

						33:		ElementAssembled:=TRUE;
								IF NOT RobotInInitialPositionBlockLow
								THEN Bit1Input:=TRUE; Bit0Input:=TRUE; (*RobotInInitialPosition:=TRUE;*)
								END_IF;
								State:=0;

	END_CASE;



2:	CASE CaseBlack OF

							(*Preleva la base nera*)
						20:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10; (*SCENDI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition>=170
								THEN	RobotGoVerticalPosition:=170;
										CaseBlack:=21;
								END_IF;

						21:		EndEffectorPosition:=EndEffectorPosition+1; (*CHIUDI END-EFFECTOR*)
								IF EndEffectorPosition>=3
								THEN 	EndEffectorPosition:=3;
										CaseBlack:=22;
								END_IF;

						22:		ElementInEndEffector:=TRUE; (*MOSTRA LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorBlack:=ElementStation1RobotBlack;
								ElementInEndEffectorSilver:=ElementStation1RobotSilver;
								ElementInEndEffectorRed:=ElementStation1RobotRed;
								ElementInEndEffectorOverturned:=ElementStation1RobotOverturned;
								IF ElementInEndEffectorOverturned
								THEN ElementInEndEffectorO:='O';
								ELSE ElementInEndEffectorO:='';
								END_IF;
								CaseBlack:=23;

						23:		ElementStation1RobotCharged:=FALSE;
								CaseBlack:=24;

						24:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10; (*SALI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN	RobotGoVerticalPosition:=0;
										CaseBlack:=25;
								END_IF;
								(*RESETTO LE INFORMAZIONI ALLA STAZIONE1 PRIMA DI METTERE A FALSE AvailableLoadForRobot IN MODO DA AVERE LE GIUSTE INFORMAZIONI SUL PEZZO SUCCESSIVO*)
						25:		ElementStation1RobotRed:=FALSE;
								ElementStation1RobotBlack:=FALSE;
								ElementStation1RobotSilver:=FALSE;
								ElementStation1RobotOverturned:=FALSE;
								IF NOT AvailableLoadForRobotBlockHigh
								THEN AvailableLoadForRobot:=FALSE; (*RESET DEL SENSORE DI PRESENZA NELLA STAZIONE1, A QUESTO PUNTO LA GIOSTRA PUO' MANDARE UN'ALTRO PEZZO*)
								END_IF;
								CaseBlack:=26;

						26 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10; (*MUOVITI ORIZZONTALMENTE VERSO DESTRA*)
								END_IF;
								IF RobotGoHorizontalPosition>=100
								THEN	RobotGoHorizontalPosition:=100;
										CaseBlack:=27;
								END_IF;

						27 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10; (*SCENDI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition>=190
								THEN 	RobotGoVerticalPosition:=190;
										CaseBlack:=28;
								END_IF;

						28 :	EndEffectorPosition:=EndEffectorPosition-1; (*APRI END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN	EndEffectorPosition:=0;
										CaseBlack:=29;
								END_IF;

						29:		ElementAssemblyCharged:=TRUE; (*MOSTRA LA BASE NELLA STAZIONE DI ASSEMBLAGGIO*)
								ElementAssemblyBlack:=ElementInEndEffectorBlack;
								ElementAssemblySilver:=ElementInEndEffectorSilver;
								ElementAssemblyRed:=ElementInEndEffectorRed;
								CaseBlack:=30;


						30 :	ElementInEndEffector:=FALSE; (*NASCONDI LA BASE SULL'END-EFFECTOR*)
								ElementInEndEffectorBlack:=FALSE;
								ElementInEndEffectorSilver:=FALSE;
								ElementInEndEffectorRed:=FALSE;
								ElementInEndEffectorOverturned:=FALSE;
								IF NOT RobotInAssemblyUnitBlockLow AND NOT RobotInInitialPositionBlockHigh
								THEN Bit3Input:=TRUE; Bit2Input:=FALSE; Bit1Input:=FALSE; Bit0Input:=FALSE; (*1000 = RobotInAssemblyUnit*)
								END_IF;
								CaseBlack:=31;

						31:		IF (NOT Bit3Output AND Bit2Output AND Bit1Output AND Bit0Output) AND NOT RobotGoToPistonHouseBlockLow  (*0111 = Prendi Pistone*)
								THEN CaseBlack:=32;
								ELSE CaseBlack:=31;
								END_IF;

						32 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=-90
								THEN	RobotGoVerticalPosition:=-90;
										CaseBlack:=33;
								END_IF;

							(*CHIUDI END-EFFECTOR PER POTER PASSARE TRA LE DUE TORRETTE DEI PISTONI*)
						33 :	EndEffectorPosition:=EndEffectorPosition+1;
								IF EndEffectorPosition>=20
								THEN 	EndEffectorPosition:=20;
										CaseBlack:=34;
								END_IF;

						34 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=280
								THEN 	RobotGoHorizontalPosition:=280;
										CaseBlack:=35;
								END_IF;

						35 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=230
								THEN	RobotGoVerticalPosition:=230;
										CaseBlack:=36;
								END_IF;

							(*CHIUDI END-EFFECTOR PER PRESA PISTONE*)
						36 :	EndEffectorPosition:=EndEffectorPosition+2;
								IF EndEffectorPosition>=25
								THEN 	EndEffectorPosition:=25;
										CaseBlack:=37;
								END_IF;

						37:		IF PistonSilverExtract AND PistonSelectorPosition=-20(*NOT PistonSelectorIsOnTheRightBlockHigh AND NOT PistonSelectorIsOnTheRightBlockLow AND NOT PistonSelectorIsOnTheLeftBlockHigh AND NOT PistonSelectorIsOnTheLeftBlockLow AND NOT PistonSelectorGoOnTheRightBlockHigh AND NOT PistonSelectorGoOnTheRightBlockLow AND NOT PistonSelectorGoOnTheLeftBlockHigh AND NOT PistonSelectorGoOnTheLeftBlockLow*)
								THEN	PistonSilverInEndEffector:=TRUE; (*RENDO VISIBILE IL PISTONE GRIGIO NELL'END-EFFECTOR*)
										PistonSilverExtract:=FALSE; (*RENDO INVISIBILE IL PISTONE GRIGIO NEL MAGAZZINO*)
										PistonSilverNumber:=PistonSilverNumber-1;
								END_IF;
								IF NOT RobotInPistonWarehouseBlockLow
								THEN Bit0Input:=TRUE; (*RobotInPistonWarehouse:=TRUE;*)
								END_IF;
								CaseBlack:=38;

						38:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow   (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseBlack:=39;
								ELSE CaseBlack:=38;
								END_IF;

						39 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInPistonWarehouseBlockHigh
								THEN Bit0Input:=FALSE; (*RobotInPistonWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-90
								THEN	RobotGoVerticalPosition:=-90;
										CaseBlack:=40;
								END_IF;

						40 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseBlack:=41;
								END_IF;

						41 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=185
								THEN	RobotGoVerticalPosition:=185;
										CaseBlack:=42;
								END_IF;

						42 :	EndEffectorPosition:=EndEffectorPosition-2; (*APRO END-EFFECTOR*)
								IF EndEffectorPosition<=16
								THEN 	EndEffectorPosition:=16;
										PistonBlackAssembly:=PistonBlackInEndEffector;
										PistonSilverAssembly:=PistonSilverInEndEffector;
										CaseBlack:=43;
								END_IF;

						43 :	PistonBlackInEndEffector:=FALSE;
								PistonSilverInEndEffector:=FALSE;
								IF NOT RobotInAssemblyUnitBlockLow
								THEN Bit3Input:=TRUE;  (*RobotInAssemblyUnit:=TRUE;*)
								END_IF;
								CaseBlack:=44;

						44:		IF (NOT Bit3Output AND Bit2Output AND Bit1Output  AND NOT Bit0Output) AND NOT RobotGoToSpringHouseBlockLow   (*0110 = RobotGoToSpringHouse*)
								THEN CaseBlack:=45;
								ELSE CaseBlack:=44;
								END_IF;

						45 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=170
								THEN	RobotGoVerticalPosition:=170;
										CaseBlack:=46;
								END_IF;

						46 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=170
								THEN 	RobotGoHorizontalPosition:=170;
										CaseBlack:=47;
								END_IF;

						47 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=245
								THEN	RobotGoVerticalPosition:=245;
										CaseBlack:=48;
								END_IF;

						48 :	EndEffectorPosition:=EndEffectorPosition+2; (*CHIUDI END-EFFECTOR PER PRESA MOLLA*)
								IF EndEffectorPosition>=20
								THEN 	EndEffectorPosition:=20;
										CaseBlack:=49;
								END_IF;

						49:		IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh AND NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockLow AND NOT ToExtractSpringInAssemblyStationBlockLow AND NOT ToExtractSpringInAssemblyStationBlockHigh
								THEN 	SpringInEndEffector:=TRUE; (*RENDO VISIBILE LA MOLLA NELL'END-EFFECTOR*)
										SpringExtract:=FALSE; (*RENDO INVISIBILE LA MOLLA NEL MAGAZZINO*)
								END_IF;
								IF NOT RobotInSpringWarehouseBlockLow
								THEN Bit1Input:=TRUE; (*RobotInSpringWarehouse:=TRUE;*)
								END_IF;
								CaseBlack:=50;

						50:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow  (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseBlack:=51;
								ELSE CaseBlack:=50;
								END_IF;

						51 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInSpringWarehouseBlockHigh
								THEN Bit1Input:=FALSE; (*RobotInSpringWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=140
								THEN	RobotGoVerticalPosition:=140;
										CaseBlack:=52;
								END_IF;

						52 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseBlack:=53;
								END_IF;

						53 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=180
								THEN	RobotGoVerticalPosition:=180;
										CaseBlack:=54;
								END_IF;

						54 :	EndEffectorPosition:=EndEffectorPosition-2; (*APERTURA END-EFFECTOR*)
								IF EndEffectorPosition<=16
								THEN 	EndEffectorPosition:=16;
										CaseBlack:=55;
								END_IF;

						55:		IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh AND NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockLow AND NOT ToExtractSpringInAssemblyStationBlockLow AND NOT ToExtractSpringInAssemblyStationBlockHigh
								THEN 	SpringInEndEffector:=FALSE; (*RENDO INVISIBILE LA MOLLA NELL'END-EFFECTOR*)
										SpringAssembly:=TRUE; (*RENDO VISIBILE LA MOLLA DENTRO LA BASE NELLA STAZIONE DI ASSEMBLAGGIO*)
								END_IF;
								IF NOT RobotInAssemblyUnitBlockLow
								THEN Bit3Input:=TRUE; (*1000 = RobotInAssemblyUnit*)
								END_IF;
								CaseBlack:=56;

						56:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND NOT Bit0Output) AND NOT RobotGoToCoverHouseBlockLow  (*1000 = RobotGoToCoverHouse*)
								THEN CaseBlack:=57;
								ELSE CaseBlack:=56;
								END_IF;

						57 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseBlack:=58;
								END_IF;

						58 :	EndEffectorPosition:=EndEffectorPosition-2; (*APERTURA END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN 	EndEffectorPosition:=0;
										CaseBlack:=59;
								END_IF;

						59 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=500
								THEN 	RobotGoHorizontalPosition:=500;
										CaseBlack:=60;
								END_IF;

						60 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=250
								THEN	RobotGoVerticalPosition:=250;
										CaseBlack:=61;
								END_IF;

						61 :	EndEffectorPosition:=EndEffectorPosition+1; (*CHIUDI END-EFFECTOR PER PRESA MOLLA*)
								IF EndEffectorPosition>=3
								THEN 	EndEffectorPosition:=3;
										CaseBlack:=62;
								END_IF;

						62:		IF NOT ToExtractCoverInAssemblyStationForwardBlockLow AND NOT Count3
								THEN 	CoverInEndEffector:=TRUE; (*RENDO VISIBILE IL COPERCHIO NEL END-EFFECTOR*)
										CoverExtract:=FALSE; (*RENDO INVISIBILE IL COPERCHIO NEL MAGAZZINO*)
										IF ToExtractCoverInAssemblyStationForwardBlockHigh OR ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh
										THEN Count3:=TRUE; (*Count3 � un contatore, necessario per far s� che in caso di bloccaggio alto venga visualizzato il primo coperchio estratto e non i successivi*)
										END_IF;
								END_IF;
								IF NOT RobotInCoverWarehouseBlockLow
								THEN Bit2Input:=TRUE; (*RobotInCoverWarehouse:=TRUE;*)
								END_IF;
								CaseBlack:=63;

						63:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow   (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseBlack:=64;
								ELSE CaseBlack:=63;
								END_IF;

						64 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInCoverWarehouseBlockHigh
								THEN Bit2Input:=FALSE; (*RobotInCoverWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseBlack:=65;
								END_IF;

						65 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseBlack:=66;
								END_IF;

						66 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=190
								THEN	RobotGoVerticalPosition:=190;
										IF NOT RobotInAssemblyUnitBlockLow
										THEN Bit3Input:=TRUE; (*1000 = RobotInAssemblyUnit*)
										END_IF;
										CaseBlack:=67;
								END_IF;
						(*Per portare il pezzo nella stazione pezzi finiti � sufficiente dare il comando di ritorno alla posizione iniziale*)
						67:		IF (NOT Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotGoToInitialPositionBlockLow    (*0001 = RobotGoToInitialPosition*)
								THEN CaseBlack:=68;
								ELSE CaseBlack:=67;
								END_IF;

						68 :	ElementInEndEffectorRed:=ElementAssemblyRed;
								ElementInEndEffectorBlack:=ElementAssemblyBlack;
								ElementInEndEffectorSilver:=ElementAssemblySilver;
								ElementInEndEffector:=TRUE;
								PistonBlackInEndEffector:=PistonBlackAssembly;
								PistonSilverInEndEffector:=PistonSilverAssembly;
								CaseBlack:=69;

						69:		CoverAssembly:=FALSE;
								SpringAssembly:=FALSE;
								PistonBlackAssembly:=FALSE;
								PistonSilverAssembly:=FALSE;
								ElementAssemblyBlack:=FALSE;
								ElementAssemblySilver:=FALSE;
								ElementAssemblyRed:=FALSE;
								CaseBlack:=70;

						70:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10; (*SALI IN DIREZIONE VERTICALE*)
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN 	RobotGoVerticalPosition:=0;
										CaseBlack:=71;
								END_IF;

						71: 	CanText:='Pezzi Finiti'; (*SCRITTA "Pezzi Finiti" ALL'INTERNO DEL CESTO*)
								CanColour:=TRUE; (* CESTO VERDE*)
								CanDiscard:=TRUE; (*VARIABILE DI VISUALIZZAZIONE: rende visibile il cesto contenente i pezzi finiti*)
								ElementToDiscard:=TRUE;	(*MOSTRA IL PEZZO CHE FINIRA' NELLA STAZIONE DI SCARTO*)
								ElementToDiscardO:='';
								ElementToDiscardRed:=ElementInEndEffectorRed;
								ElementToDiscardBlack:=ElementInEndEffectorBlack;
								ElementToDiscardSilver:=ElementInEndEffectorSilver;
								IF CoverInEndEffector
								THEN CoverToDiscard:=TRUE; (*MOSTRA IL COPERCHIO*)
								END_IF;
								PistonBlackToDiscard:=PistonBlackInEndEffector;
								PistonSilverToDiscard:=PistonSilverInEndEffector;
								CaseBlack:=72;

						72:		EndEffectorPosition:=EndEffectorPosition-1; (*APRI END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN 	EndEffectorPosition:=0;
										CaseBlack:=73;
								END_IF;

						73: 	ElementInEndEffector:=FALSE; (*RENDO INVISIBILE LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorRed:=FALSE; (*RESET DELLE BASI NELL'END-EFFECTOR, NON L'HO FATTO NELL'IF PRECEDENTE PERCHE' AVREI RESO INVISIBILI LE BASI NELLA STAZIONE SCARTI*)
								ElementInEndEffectorBlack:=FALSE;
								ElementInEndEffectorSilver:=FALSE;
								ElementInEndEffectorOverturned:=FALSE;
								CoverInEndEffector:=FALSE;
								PistonBlackInEndEffector:=FALSE;
								PistonSilverInEndEffector:=FALSE;
								CaseBlack:=74;

						74:		ElementToDiscardPosition:=ElementToDiscardPosition+4; (*lo scarto scende fino a finire nel cestino*)
								IF ElementToDiscardPosition>=80
								THEN 	ElementToDiscardPosition:=80;
										CaseBlack:=75;
								END_IF;

						75:		IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=0
								THEN 	RobotGoHorizontalPosition:=0;
										CaseBlack:=76;
								END_IF;

						76:		ElementAssembled:=TRUE;
								IF NOT RobotInInitialPositionBlockLow
								THEN Bit1Input:=TRUE; Bit0Input:=TRUE; (*RobotInInitialPosition:=TRUE; *)
								END_IF;
								State:=0;
	END_CASE;




3:	CASE CaseRedSilver OF

							(*Preleva la base rossa-argento*)
						20:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10; (*SCENDI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition>=170
								THEN	RobotGoVerticalPosition:=170;
										CaseRedSilver:=21;
								END_IF;

						21:		EndEffectorPosition:=EndEffectorPosition+1; (*CHIUDI END-EFFECTOR*)
								IF EndEffectorPosition>=3
								THEN 	EndEffectorPosition:=3;
										CaseRedSilver:=22;
								END_IF;

						22:		ElementInEndEffector:=TRUE; (*MOSTRA LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorBlack:=ElementStation1RobotBlack;
								ElementInEndEffectorSilver:=ElementStation1RobotSilver;
								ElementInEndEffectorRed:=ElementStation1RobotRed;
								ElementInEndEffectorOverturned:=ElementStation1RobotOverturned;
								IF ElementInEndEffectorOverturned
								THEN ElementInEndEffectorO:='O';
								ELSE ElementInEndEffectorO:='';
								END_IF;
								CaseRedSilver:=23;

						23:		ElementStation1RobotCharged:=FALSE;
								CaseRedSilver:=24;

						24:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10; (*SALI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN	RobotGoVerticalPosition:=0;
										CaseRedSilver:=25;
								END_IF;

						25:		ElementStation1RobotRed:=FALSE; (*RESETTO LE INFORMAZIONI ALLA STAZIONE1 PRIMA DI METTERE A FALSE AvailableLoadForRobot IN MODO DA AVERE LE GIUSTE INFORMAZIONI SUL PEZZO SUCCESSIVO*)
								ElementStation1RobotBlack:=FALSE;
								ElementStation1RobotSilver:=FALSE;
								ElementStation1RobotOverturned:=FALSE;
								IF NOT AvailableLoadForRobotBlockHigh
								THEN AvailableLoadForRobot:=FALSE; (*RESET DEL SENSORE DI PRESENZA NELLA STAZIONE1, A QUESTO PUNTO LA GIOSTRA PUO' MANDARE UN'ALTRO PEZZO*)
								END_IF;
								CaseRedSilver:=26;

						26 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10; (*MUOVITI ORIZZONTALMENTE VERSO DESTRA*)
								END_IF;
								IF RobotGoHorizontalPosition>=100
								THEN	RobotGoHorizontalPosition:=100;
										CaseRedSilver:=27;
								END_IF;

						27 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10; (*SCENDI SECONDO LA DIREZIONE VERTICALE*)
								END_IF;
								IF RobotGoVerticalPosition>=190
								THEN 	RobotGoVerticalPosition:=190;
										CaseRedSilver:=28;
								END_IF;

						28 :	EndEffectorPosition:=EndEffectorPosition-1; (*APRI END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN	EndEffectorPosition:=0;
										CaseRedSilver:=29;
								END_IF;

						29:		ElementAssemblyCharged:=TRUE; (*MOSTRA LA BASE NELLA STAZIONE DI ASSEMBLAGGIO*)
								ElementAssemblyBlack:=ElementInEndEffectorBlack;
								ElementAssemblySilver:=ElementInEndEffectorSilver;
								ElementAssemblyRed:=ElementInEndEffectorRed;
								CaseRedSilver:=30;

						30 :	ElementInEndEffector:=FALSE; (*NASCONDI LA BASE SULL'END-EFFECTOR*)
								ElementInEndEffectorBlack:=FALSE;
								ElementInEndEffectorSilver:=FALSE;
								ElementInEndEffectorRed:=FALSE;
								ElementInEndEffectorOverturned:=FALSE;
								IF NOT RobotInAssemblyUnitBlockLow AND NOT RobotInInitialPositionBlockHigh
								THEN Bit3Input:=TRUE; Bit2Input:=FALSE; Bit1Input:=FALSE; Bit0Input:=FALSE; (*1000 = RobotInAssemblyUnit*)
								END_IF;
								CaseRedSilver:=31;

						31:		IF (NOT Bit3Output AND Bit2Output AND Bit1Output AND Bit0Output) AND NOT RobotGoToPistonHouseBlockLow   (*0111 = Prendi Pistone*)
								THEN CaseRedSilver:=32;
								ELSE CaseRedSilver:=31;
								END_IF;

						32 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseRedSilver:=33;
								END_IF;

						33 :	EndEffectorPosition:=EndEffectorPosition+2; (*CHIUDI END-EFFECTOR PER POTER PASSARE TRA LE DUE TORRETTE DEI PISTONI*)
								IF EndEffectorPosition>=20
								THEN 	EndEffectorPosition:=20;
										CaseRedSilver:=34;
								END_IF;

						34 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=435
								THEN 	RobotGoHorizontalPosition:=435;
										CaseRedSilver:=35;
								END_IF;

						35 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=230
								THEN	RobotGoVerticalPosition:=230;
										CaseRedSilver:=36;
								END_IF;

						36 :	EndEffectorPosition:=EndEffectorPosition+2; (*CHIUDI END-EFFECTOR PER PRESA PISTONE*)
								IF EndEffectorPosition>=25
								THEN 	EndEffectorPosition:=25;
										CaseRedSilver:=37;
								END_IF;

						37:		IF PistonBlackExtract AND PistonSelectorPosition=20(*NOT PistonSelectorIsOnTheRightBlockHigh AND NOT PistonSelectorIsOnTheRightBlockLow AND NOT PistonSelectorIsOnTheLeftBlockHigh AND NOT PistonSelectorIsOnTheLeftBlockLow AND NOT PistonSelectorGoOnTheRightBlockHigh AND NOT PistonSelectorGoOnTheRightBlockLow AND NOT PistonSelectorGoOnTheLeftBlockHigh AND NOT PistonSelectorGoOnTheLeftBlockLow*)
								THEN	PistonBlackInEndEffector:=TRUE; (*RENDO VISIBILE IL PISTONE NERO NELL'END-EFFECTOR*)
										PistonBlackExtract:=FALSE; (*RENDO INVISIBILE IL PISTONE NERO NEL MAGAZZINO*)
										PistonBlackNumber:=PistonBlackNumber-1;
								END_IF;
								IF NOT RobotInPistonWarehouseBlockLow
								THEN Bit0Input:=TRUE; (*RobotInPistonWarehouse:=TRUE;*)
								END_IF;
								CaseRedSilver:=38;

						38:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow  (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseRedSilver:=39;
								ELSE CaseRedSilver:=38;
								END_IF;

						39 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInPistonWarehouseBlockHigh
								THEN Bit0Input:=FALSE; (*RobotInPistonWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseRedSilver:=40;
								END_IF;

						40 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseRedSilver:=41;
								END_IF;

						41 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=185
								THEN	RobotGoVerticalPosition:=185;
										CaseRedSilver:=42;
								END_IF;

						42 :	EndEffectorPosition:=EndEffectorPosition-2; (*APRO END-EFFECTOR*)
								IF EndEffectorPosition<=16
								THEN 	EndEffectorPosition:=16;
										PistonBlackAssembly:=PistonBlackInEndEffector;
										PistonSilverAssembly:=PistonSilverInEndEffector;
										CaseRedSilver:=43;
								END_IF;

						43 :	PistonBlackInEndEffector:=FALSE;
								PistonSilverInEndEffector:=FALSE;
								IF NOT RobotInAssemblyUnitBlockLow
								THEN Bit3Input:=TRUE;  (*RobotInAssemblyUnit:=TRUE;*)
								END_IF;
								CaseRedSilver:=44;

						44:		IF (NOT Bit3Output AND Bit2Output AND Bit1Output AND NOT Bit0Output) AND NOT RobotGoToSpringHouseBlockLow  (*0110 = RobotGoToSpringHouse*)
								THEN CaseRedSilver:=45;
								ELSE CaseRedSilver:=44;
								END_IF;

						45 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=170
								THEN	RobotGoVerticalPosition:=170;
										CaseRedSilver:=46;
								END_IF;

						46 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=170
								THEN 	RobotGoHorizontalPosition:=170;
										CaseRedSilver:=47;
								END_IF;

						47 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=245
								THEN	RobotGoVerticalPosition:=245;
										CaseRedSilver:=48;
								END_IF;

						48 :	EndEffectorPosition:=EndEffectorPosition+2; (*CHIUDI END-EFFECTOR PER PRESA MOLLA*)
								IF EndEffectorPosition>=20
								THEN 	EndEffectorPosition:=20;
										CaseRedSilver:=49;
								END_IF;

						49:		IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh AND NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockLow AND NOT ToExtractSpringInAssemblyStationBlockLow AND NOT ToExtractSpringInAssemblyStationBlockHigh
								THEN 	SpringInEndEffector:=TRUE; (*RENDO VISIBILE LA MOLLA NELL'END-EFFECTOR*)
										SpringExtract:=FALSE; (*RENDO INVISIBILE LA MOLLA NEL MAGAZZINO*)
								END_IF;
								IF NOT RobotInSpringWarehouseBlockLow
								THEN Bit1Input:=TRUE; (*RobotInSpringWarehouse:=TRUE;*)
								END_IF;
								CaseRedSilver:=50;

						50:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow  (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseRedSilver:=51;
								ELSE CaseRedSilver:=50;
								END_IF;

						51 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInSpringWarehouseBlockHigh
								THEN Bit1Input:=FALSE; (*RobotInSpringWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=140
								THEN	RobotGoVerticalPosition:=140;
										CaseRedSilver:=52;
								END_IF;

						52 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseRedSilver:=53;
								END_IF;

						53 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=180
								THEN	RobotGoVerticalPosition:=180;
										CaseRedSilver:=54;
								END_IF;

						54 :	EndEffectorPosition:=EndEffectorPosition-2; (*APERTURA END-EFFECTOR*)
								IF EndEffectorPosition<=16
								THEN 	EndEffectorPosition:=16;
										CaseRedSilver:=55;
								END_IF;

						55:		IF NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh AND NOT ToExtractSpringInAssemblyStationInExtensivePositionBlockLow AND NOT ToExtractSpringInAssemblyStationBlockLow AND NOT ToExtractSpringInAssemblyStationBlockHigh
								THEN 	SpringInEndEffector:=FALSE; (*RENDO INVISIBILE LA MOLLA NELL'END-EFFECTOR*)
										SpringAssembly:=TRUE; (*RENDO VISIBILE LA MOLLA DENTRO LA BASE NELLA STAZIONE DI ASSEMBLAGGIO*)
								END_IF;
								IF NOT RobotInAssemblyUnitBlockLow
								THEN Bit3Input:=TRUE; (*1000 = RobotInAssemblyUnit*)
								END_IF;
								CaseRedSilver:=56;

						56:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND NOT Bit0Output) AND NOT RobotGoToCoverHouseBlockLow  (*1000 = RobotGoToCoverHouse*)
								THEN CaseRedSilver:=57;
								ELSE CaseRedSilver:=56;
								END_IF;

						57 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseRedSilver:=58;
								END_IF;

						58 :	EndEffectorPosition:=EndEffectorPosition-2; (*APERTURA END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN 	EndEffectorPosition:=0;
										CaseRedSilver:=59;
								END_IF;

						59 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition+10;
								END_IF;
								IF RobotGoHorizontalPosition>=500
								THEN 	RobotGoHorizontalPosition:=500;
										CaseRedSilver:=60;
								END_IF;

						60 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=250
								THEN	RobotGoVerticalPosition:=250;
										CaseRedSilver:=61;
								END_IF;

						61 :	EndEffectorPosition:=EndEffectorPosition+1; (*CHIUDI END-EFFECTOR PER PRESA MOLLA*)
								IF EndEffectorPosition>=3
								THEN 	EndEffectorPosition:=3;
										CaseRedSilver:=62;
								END_IF;

						62:		IF NOT ToExtractCoverInAssemblyStationForwardBlockLow AND NOT Count3
								THEN 	CoverInEndEffector:=TRUE; (*RENDO VISIBILE IL COPERCHIO NEL END-EFFECTOR*)
										CoverExtract:=FALSE; (*RENDO INVISIBILE IL COPERCHIO NEL MAGAZZINO*)
										IF ToExtractCoverInAssemblyStationForwardBlockHigh OR ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh
										THEN Count3:=TRUE;
										END_IF;
								END_IF;
								IF NOT RobotInCoverWarehouseBlockLow
								THEN Bit2Input:=TRUE; (*RobotInCoverWarehouse:=TRUE;*)
								END_IF;
								CaseRedSilver:=63;

						63:		IF (Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotTakeCurrentLoadToAssemblyBlockLow   (*1001 = TakeCurrentLoadToAssemblyStation*)
								THEN CaseRedSilver:=64;
								ELSE CaseRedSilver:=63;
								END_IF;

						64 :	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF NOT RobotInCoverWarehouseBlockHigh
								THEN Bit2Input:=FALSE; (*RobotInCoverWarehouse:=FALSE;*)
								END_IF;
								IF RobotGoVerticalPosition<=-110
								THEN	RobotGoVerticalPosition:=-110;
										CaseRedSilver:=65;
								END_IF;

						65 :	IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=100
								THEN 	RobotGoHorizontalPosition:=100;
										CaseRedSilver:=66;
								END_IF;

						66 : 	IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=190
								THEN	RobotGoVerticalPosition:=190;
										IF NOT RobotInAssemblyUnitBlockLow
										THEN Bit3Input:=TRUE; (*1000 = RobotInAssemblyUnit*)
										END_IF;
										CaseRedSilver:=67;
								END_IF;
						(*Per portare il pezzo nella stazione pezzi finiti � sufficiente dare il comando di ritorno alla posizione iniziale*)
						67:		IF (NOT Bit3Output AND NOT Bit2Output AND NOT Bit1Output AND Bit0Output) AND NOT RobotGoToInitialPositionBlockLow  (*0001 = RobotGoToInitialPosition*)
								THEN CaseRedSilver:=68;
								ELSE CaseRedSilver:=67;
								END_IF;

						68 :	ElementInEndEffectorRed:=ElementAssemblyRed;
								ElementInEndEffectorBlack:=ElementAssemblyBlack;
								ElementInEndEffectorSilver:=ElementAssemblySilver;
								ElementInEndEffector:=TRUE;
								PistonBlackInEndEffector:=PistonBlackAssembly;
								PistonSilverInEndEffector:=PistonSilverAssembly;
								CaseRedSilver:=69;

						69:		CoverAssembly:=FALSE;
								SpringAssembly:=FALSE;
								PistonBlackAssembly:=FALSE;
								PistonSilverAssembly:=FALSE;
								ElementAssemblyBlack:=FALSE;
								ElementAssemblySilver:=FALSE;
								ElementAssemblyRed:=FALSE;
								CaseRedSilver:=70;

						70:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10; (*SALI IN DIREZIONE VERTICALE*)
								END_IF;
								IF NOT RobotInAssemblyUnitBlockHigh
								THEN Bit3Input:=FALSE; (*RobotInAssemblyUnit:=FALSE*)
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN 	RobotGoVerticalPosition:=0;
										CaseRedSilver:=71;
								END_IF;

						71: 	CanText:='Pezzi Finiti'; (*SCRITTA "Pezzi Finiti" ALL'INTERNO DEL CESTO*)
								CanColour:=TRUE; (* CESTO VERDE*)
								CanDiscard:=TRUE; (*VARIABILE DI VISUALIZZAZIONE: rende visibile il cesto contenente i pezzi finiti*)
								ElementToDiscard:=TRUE;	(*MOSTRA IL PEZZO CHE FINIRA' NELLA STAZIONE DI SCARTO*)
								ElementToDiscardO:='';
								ElementToDiscardRed:=ElementInEndEffectorRed;
								ElementToDiscardBlack:=ElementInEndEffectorBlack;
								ElementToDiscardSilver:=ElementInEndEffectorSilver;
								IF CoverInEndEffector
								THEN CoverToDiscard:=TRUE; (*MOSTRA IL COPERCHIO*)
								END_IF;
								PistonBlackToDiscard:=PistonBlackInEndEffector;
								PistonSilverToDiscard:=PistonSilverInEndEffector;
								CaseRedSilver:=72;

						72:		EndEffectorPosition:=EndEffectorPosition-1; (*APRI END-EFFECTOR*)
								IF EndEffectorPosition<=0
								THEN 	EndEffectorPosition:=0;
										CaseRedSilver:=73;
								END_IF;

						73: 	ElementInEndEffector:=FALSE; (*RENDO INVISIBILE LA BASE NELL'END-EFFECTOR*)
								ElementInEndEffectorRed:=FALSE; (*RESET DELLE BASI NELL'END-EFFECTOR, NON L'HO FATTO NELL'IF PRECEDENTE PERCHE' AVREI RESO INVISIBILI LE BASI NELLA STAZIONE SCARTI*)
								ElementInEndEffectorBlack:=FALSE;
								ElementInEndEffectorSilver:=FALSE;
								ElementInEndEffectorOverturned:=FALSE;
								CoverInEndEffector:=FALSE;
								PistonBlackInEndEffector:=FALSE;
								PistonSilverInEndEffector:=FALSE;
								CaseRedSilver:=74;

						74:		ElementToDiscardPosition:=ElementToDiscardPosition+4; (*lo scarto scende fino a finire nel cestino*)
								IF ElementToDiscardPosition>=80
								THEN 	ElementToDiscardPosition:=80;
										CaseRedSilver:=75;
								END_IF;

						75:		IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=0
								THEN 	RobotGoHorizontalPosition:=0;
										CaseRedSilver:=76;
								END_IF;

						76:		ElementAssembled:=TRUE;
								IF NOT RobotInInitialPositionBlockLow
								THEN Bit1Input:=TRUE; Bit0Input:=TRUE; (*RobotInInitialPosition:=TRUE; *)
								END_IF;
								State:=0;

	END_CASE;

4 :	CASE InitialPosition OF

						20:		IF RobotGoVerticalPosition<0
								THEN InitialPosition:=21;
								ELSE InitialPosition:=22;
								END_IF;

						21:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition+10;
								END_IF;
								IF RobotGoVerticalPosition>=0
								THEN 	RobotGoVerticalPosition:=0;
										InitialPosition:=23;
								END_IF;

						22:		IF NOT RobotEngineBlockLow
								THEN RobotGoVerticalPosition:=RobotGoVerticalPosition-10;
								END_IF;
								IF RobotGoVerticalPosition<=0
								THEN 	RobotGoVerticalPosition:=0;
										InitialPosition:=23;
								END_IF;

						23:		IF NOT RobotEngineBlockLow
								THEN RobotGoHorizontalPosition:=RobotGoHorizontalPosition-10;
								END_IF;
								IF RobotGoHorizontalPosition<=0
								THEN 	RobotGoHorizontalPosition:=0;
										EndEffectorPosition:=EndEffectorPosition-1; (*CHIUDI END-EFFECTOR*)
										IF EndEffectorPosition<=0
										THEN 	EndEffectorPosition:=0;
												IF NOT RobotInInitialPositionBlockLow
												THEN Bit1Input:=TRUE; Bit0Input:=TRUE; (*RobotInInitialPosition:=TRUE; *)
												END_IF;
												State:=0;
										END_IF;

								END_IF;
	END_CASE;

END_CASE;

(*Count3 viene utilizzato nello State=62...� un contatore utilizzato per visualizzare correttemente il coperchio in magazzino in caso di bloccaggio alto del pistone*)
IF NOT ToExtractCoverInAssemblyStationForwardBlockHigh AND NOT ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh
THEN 	Count3:=FALSE;
END_IF;


(*CILINDRO DI BLOCCAGGIO(Normalmente Aperto) PEZZO NELLA STAZIONE DI ASSEMBLAGGIO*)
IF FLAGBlockingCylinderForwardInAssemblyStationBlockHigh OR (BlockingCylinderForwardInAssemblyStation AND NOT BlockingCylinderForwardInAssemblyStationBlockLow)
THEN 	CylinderPositionInAssemblyUnit:=CylinderPositionInAssemblyUnit-1;
		IF BlockingCylinderForwardInAssemblyStationBlockHigh
		THEN FLAGBlockingCylinderForwardInAssemblyStationBlockHigh:=TRUE;
		ELSE FLAGBlockingCylinderForwardInAssemblyStationBlockHigh:=FALSE;
		END_IF;
ELSE CylinderPositionInAssemblyUnit:=CylinderPositionInAssemblyUnit+1;
END_IF;

IF CylinderPositionInAssemblyUnit>=6
THEN CylinderPositionInAssemblyUnit:=6;
END_IF;

IF CylinderPositionInAssemblyUnit<=0
THEN CylinderPositionInAssemblyUnit:=0;
END_IF;

IF Remove
THEN 	ElementAssembled:=TRUE;
		ElementInEndEffector:=FALSE;
		CoverInEndEffector:=FALSE;
		PistonBlackInEndEffector:=FALSE;
		PistonSilverInEndEffector:=FALSE;
		SpringInEndEffector:=FALSE;
		ElementAssemblyBlack:=FALSE;
		ElementAssemblyRed:=FALSE;
		ElementAssemblySilver:=FALSE;
		CoverAssembly:=FALSE;
		PistonBlackAssembly:=FALSE;
		PistonSilverAssembly:=FALSE;
		SpringAssembly:=FALSE;
		SpringExtract:=FALSE;
		CoverExtract:=FALSE;
		ElementStation1RobotCharged:=FALSE;
		(*FillAllWarehouses:=TRUE;*)
		AvailableLoadForRobot:=FALSE;
		IF PistonSelectorIsOnTheLeft
		THEN PistonBlackExtract:=FALSE;
		END_IF;
END_IF;               1   , ] ] �"           PlantCarico ��d	��d      ��������        :  PROGRAM PlantCarico
VAR
	EstrazioneRosso: INT:=0; (*variabile che memorizza la posizione dei pezzi rossi estratti*)
	EstrazioneSilver:INT:=0; (*variabile che memorizza la posizione dei pezzi argentati estratti*)
	EstrazioneNero: INT:=0; (*variabile che memorizza la posizione dei pezzi neri estratti*)
END_VARA   (*Per caricare gli oggetti nel magazzino, utilizzo un'istruzione "case of" per ogni pulsante -red, silver, black-*)

IF (NOT ElementOneCharged AND (Red OR Redoverturned )) THEN
	EstrazioneRosso:=1 ;(*carico il primo elemento rosso*)
END_IF;

IF (ElementOneCharged AND (Red OR Redoverturned)) THEN
	EstrazioneRosso:=2;
END_IF; (*carico il secondo elemento rosso*)

IF ( ElementTwoCharged  AND (Red OR Redoverturned) ) THEN
	EstrazioneRosso:=3;(*carico il terzo  elemento rosso*)
END_IF;

 IF (ElementThreeCharged AND (Red OR Redoverturned) ) THEN
	EstrazioneRosso:=4;  (*carico il quarto elemento rosso*)
END_IF;

IF (ElementFourCharged AND (Red OR Redoverturned) ) THEN
	EstrazioneRosso:=5; (*carico il quinto elemento rosso*)
END_IF;

IF (ElementFiveCharged AND (Red OR Redoverturned) ) THEN
	EstrazioneRosso:=6;  (*carico il sesto elemento rosso*)
END_IF;

IF (ElementSixCharged AND (Red  OR Redoverturned)) THEN
	EstrazioneRosso:=7; (*carico il settimo elemento rosso*)
END_IF;

IF (ElementSevenCharged AND (Red OR Redoverturned)) THEN
	EstrazioneRosso:=8;  (*carico l'ottavo elemento rosso*)
END_IF;

CASE EstrazioneRosso OF	(*Ogni stato dell'istruzione "case of" corrisponde ad una base*)

1:	ElementOneRed:=TRUE;
	EmptyWarehouse:=FALSE;
	ElementOneCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementOneOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

2:	ElementTwoRed:=TRUE;
	ElementTwoCharged:=TRUE;
	Red:=FALSE;
	IF   Redoverturned THEN
		ElementTwoOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

3:	ElementThreeRed:=TRUE;
	ElementThreeCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementThreeOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

4:	ElementFourRed:=TRUE;
	ElementFourCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementFourOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

5:	ElementFiveRed:=TRUE;
	ElementFiveCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementFiveOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

6:	ElementSixRed:=TRUE;
	ElementSixCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementSixOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

7:	ElementSevenRed:=TRUE;
	ElementSevenCharged:=TRUE;
	Red:=FALSE;
      IF Redoverturned THEN
		ElementSevenOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;

8:	ElementEightRed:=TRUE;
	ElementEightCharged:=TRUE;
	Red:=FALSE;
	IF Redoverturned THEN
		ElementEightOverturned:=TRUE;
	END_IF;
	Redoverturned:=FALSE;
	EstrazioneRosso:=0;
END_CASE;


IF (NOT ElementOneCharged AND (Silver OR Silveroverturned OR Silvershort)) THEN
	EstrazioneSilver:=1 ;
END_IF;

IF (ElementOneCharged  AND (Silver OR Silveroverturned OR Silvershort ) ) THEN
	EstrazioneSilver:=2;
END_IF;

IF ( ElementTwoCharged  AND (Silver OR Silveroverturned  OR Silvershort) ) THEN
	EstrazioneSilver:=3;
END_IF;

 IF (ElementThreeCharged AND (Silver OR Silveroverturned  OR Silvershort) ) THEN
	EstrazioneSilver:=4;
END_IF;

IF (ElementFourCharged AND (Silver OR Silveroverturned  OR Silvershort) ) THEN
	EstrazioneSilver:=5;
END_IF;

IF (ElementFiveCharged AND (Silver OR Silveroverturned  OR Silvershort) ) THEN
	EstrazioneSilver:=6;
END_IF;

IF (ElementSixCharged AND (Silver OR Silveroverturned  OR Silvershort)  ) THEN
	EstrazioneSilver:=7;
END_IF;

IF (ElementSevenCharged AND (Silver OR Silveroverturned  OR Silvershort) ) THEN
		EstrazioneSilver:=8;
END_IF;


CASE EstrazioneSilver OF

1:	ElementOneSilver:=TRUE;
	ElementOneCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN  (* cilindro capovolto *)
		ElementOneOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN  (* cilindro corto *)
		ElementOneShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

2:	ElementTwoSilver:=TRUE;
	ElementTwoCharged:=TRUE;
	Silver:=FALSE;
	IF 	Silveroverturned THEN
		ElementTwoOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementTwoShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

3:	ElementThreeSilver:=TRUE;
	ElementThreeCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementThreeOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementThreeShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

4:	ElementFourSilver:=TRUE;
	ElementFourCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementFourOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementFourShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

5:	ElementFiveSilver:=TRUE;
	ElementFiveCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementFiveOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementFiveShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

6:	ElementSixSilver:=TRUE;
	ElementSixCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementSixOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementSixShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

7:	ElementSevenSilver:=TRUE;
	ElementSevenCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementSevenOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementSevenShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;

8:	ElementEightSilver:=TRUE;
	ElementEightCharged:=TRUE;
	Silver:=FALSE;
	IF Silveroverturned THEN
		ElementEightOverturned:=TRUE;
	END_IF;
	Silveroverturned:=FALSE;
	IF Silvershort THEN
		ElementEightShort:=TRUE;
	END_IF;
	Silvershort:=FALSE;
	EstrazioneSilver:=0;
END_CASE;

IF (NOT ElementOneCharged AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=1 ;
END_IF;

IF (ElementOneCharged  AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=2;
END_IF;

IF ( ElementTwoCharged  AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=3;
END_IF;

 IF (ElementThreeCharged AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=4;
END_IF;

IF (ElementFourCharged AND (Black OR Blackoverturned)  ) THEN
	EstrazioneNero:=5;
END_IF;

IF (ElementFiveCharged AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=6;
END_IF;
IF (ElementSixCharged AND (Black OR Blackoverturned)  ) THEN
	EstrazioneNero:=7;
END_IF;

IF (ElementSevenCharged AND (Black OR Blackoverturned) ) THEN
	EstrazioneNero:=8;
END_IF;


CASE EstrazioneNero OF

1:	ElementOneBlack:=TRUE;
	ElementOneCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementOneOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

2:	ElementTwoBlack:=TRUE;
	ElementTwoCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementTwoOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

3:	ElementThreeBlack:=TRUE;
	ElementThreeCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementThreeOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

4:	ElementFourBlack:=TRUE;
	ElementFourCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementFourOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

5:	ElementFiveBlack:=TRUE;
	ElementFiveCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementFiveOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

6:	ElementSixBlack:=TRUE;
	ElementSixCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementSixOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

7:	ElementSevenBlack:=TRUE;
	ElementSevenCharged:=TRUE;
	Black:=FALSE;
	IF Blackoverturned THEN
		ElementSevenOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

8:	ElementEightBlack:=TRUE;
	ElementEightCharged:=TRUE;
	Black:=FALSE;
	IF	Blackoverturned THEN
		ElementEightOverturned:=TRUE;
	END_IF;
	Blackoverturned:=FALSE;
	EstrazioneNero:=0;

END_CASE;               �  , � � (�           PlantLavorazione ��d	��d      ��������        �  PROGRAM PlantLavorazione
VAR
	Set:BOOL:=TRUE;
	FLAG: BOOL := TRUE;
	ElementInControlOverturned: BOOL;
	FLAGRotaryTableMotorBlockHigh: BOOL;
	FLAGToLowerCylinderToInspectLoadBlockHigh: BOOL;
	FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh: BOOL;
	FLAGToLowerDrillingUnitBlockHigh: BOOL;
	FLAGToLiftDrillingUnitBlockHigh: BOOL;
	FLAGExpellingLeverActiveBlockHigh: BOOL;
END_VAR��  (*STAZIONE DI LAVORAZIONE - Gestione della Tavola Rotante*)

(*Posizione iniziale della tavola selezionabile mediante pulsante, per simulare una inizializzazione da posizione indefinita*)
IF FLAG
THEN 	FLAG:=FALSE;
		RotaryTableVisual:=RotaryVisIn;	(*variabile di posizione "visuale"*)
		RotaryTablePosition:=RotaryVisIn-30;
END_IF;

(*Attivando il comando di rotazione della tavola rotante, viene incrementata la psozione della tavola *)
IF NOT RotaryTableMotorBlockLow
THEN	IF RotaryTableMotor OR (FLAGRotaryTableMotorBlockHigh AND EmergencyStop)
		THEN	RotaryTablePosition:=RotaryTablePosition+2;
				RotaryTableVisual:=RotaryTableVisual+2;
				IF RotaryTableMotorBlockHigh
				THEN FLAGRotaryTableMotorBlockHigh:=TRUE;
				ELSE FLAGRotaryTableMotorBlockHigh:=FALSE;
				END_IF;

				(*    La posizione viene incrementata ad ogni ciclo del programma, quindi potrebbe verificarsi il superamento della posizione desiderata,
					con le conseguenze di un errato funzionamento del sistema.
                                  A questo proposito ogni volta che viene spento il motore della tavola, vengono settati i corretti valori di posizione*)
		ELSE	IF RotaryTablePosition>357 OR (RotaryTablePosition>-3 AND RotaryTablePosition<3)
				THEN RotaryTablePosition:=0;
						RotaryTableVisual:=30;
				END_IF;
				IF RotaryTablePosition>57 AND RotaryTablePosition<63
				THEN RotaryTablePosition:=60;
						RotaryTableVisual:=90;
				END_IF;
				IF RotaryTablePosition>117 AND RotaryTablePosition<123
				THEN RotaryTablePosition:=120;
						RotaryTableVisual:=150;
				END_IF;
				IF RotaryTablePosition>177 AND RotaryTablePosition<183
				THEN RotaryTablePosition:=180;
						RotaryTableVisual:=210;
				END_IF;
				IF RotaryTablePosition>237 AND RotaryTablePosition<243
				THEN RotaryTablePosition:=240;
						RotaryTableVisual:=270;
				END_IF;
				IF RotaryTablePosition>297 AND RotaryTablePosition<303
				THEN RotaryTablePosition:=300;
						RotaryTableVisual:=330;
				END_IF;
		END_IF;
END_IF;

(*La variabile di spostamento della tavola torna a 0 ogni volta che quest'ultima si muove di 360�*)
IF RotaryTablePosition=360
THEN	RotaryTablePosition:=0;
END_IF;

IF RotaryTableVisual=360
THEN	RotaryTableVisual:=0;
END_IF;
(*Quando una base transita tra stazione di verifica e stazione di lavorazione (passando per la slitta), viene attivata la variabile ElementAirVsRotaryTable.*)
(*Quindi, quando � attiva la variabile ElementAirVsRotaryTable, posso attivare il sensore di presenza che indica l'arrivo di una base nella giostra*)
IF (ElementAirVsRotaryTable AND NOT AvailableLoadForWorkingStationBlockLow) OR AvailableLoadForWorkingStationBlockHigh
THEN	AvailableLoadForWorkingStation:=TRUE;
ELSE   AvailableLoadForWorkingStation:=FALSE;
END_IF;

(*Quando la tavola inizia a muoversi si azzera il sensore di presenza pezzo nella prima stazione*)
IF ((RotaryTablePosition>=1 AND RotaryTablePosition<=2) OR (RotaryTablePosition>=61 AND RotaryTablePosition<=62) OR (RotaryTablePosition>=121 AND RotaryTablePosition<=122) OR (RotaryTablePosition>=181 AND RotaryTablePosition<=182) OR (RotaryTablePosition>=241 AND RotaryTablePosition<=242)) AND NOT AvailableLoadForWorkingStationBlockHigh
THEN   ElementAirVsRotaryTable:=FALSE;
END_IF;

(*Ogni volta che la tavola compie 60�, si attiva il sensore di allineamento della tavola*)
IF AlignementRotaryTableWithPositioningsBlockHigh OR (NOT AlignementRotaryTableWithPositioningsBlockLow AND ((RotaryTablePosition>=-2 AND RotaryTablePosition<=2) OR (RotaryTablePosition>=58 AND RotaryTablePosition<=62) OR (RotaryTablePosition>=118 AND RotaryTablePosition<=122) OR (RotaryTablePosition>=178 AND RotaryTablePosition<=182) OR (RotaryTablePosition>=238 AND RotaryTablePosition<=242) OR (RotaryTablePosition>=298 AND RotaryTablePosition<=302)))
THEN	AlignementRotaryTableWithPositionings:=TRUE;	(*Si attiva il sensore relativo alla posizione della tavola*)
ELSE   AlignementRotaryTableWithPositionings:=FALSE;
END_IF;

(*Utilizzo Alligneed per mettere a true i sensori Available... anche in caso di Fault di AlignementRotary...*)
IF (RotaryTablePosition>=-2 AND RotaryTablePosition<=2) OR (RotaryTablePosition>=58 AND RotaryTablePosition<=62) OR (RotaryTablePosition>=118 AND RotaryTablePosition<=122) OR (RotaryTablePosition>=178 AND RotaryTablePosition<=182) OR (RotaryTablePosition>=238 AND RotaryTablePosition<=242) OR (RotaryTablePosition>=298 AND RotaryTablePosition<=302)
THEN Alligneed:=TRUE;
ELSE Alligneed:=FALSE;
END_IF;

(*Reset dei sensori di controllo e foratura non appena la tavola ha ruotato abbastanza da non essere pi� allineata*)
IF NOT Alligneed
THEN 	IF NOT AvailableLoadInControlPositioningBlockHigh
		THEN AvailableLoadInControlPositioning:=FALSE;
		END_IF;
		IF NOT AvailableLoadInDrillingPositioningBlockHigh
		THEN AvailableLoadInDrillingPositioning:=FALSE;
		END_IF;
END_IF;


(*********************************************************************************************************************)

(*GESTIONE COLORE DELLE STAZIONI DELLA GIOSTRA*)

(*Reset dei colori dalla posizione di 180�, ovvero dopo l'espulsione dalla giostra*)
(*
ColorCircle[1]=NERO
ColorCircle[2]=BIANCO
ColorCircle[3]=ROSSO
ColorCircle[4]=GRIGIO
*)

IF (ElementSleighCharged OR ExpellingLeverActiveBlockLow OR FLAGExpellingLeverActiveBlockHigh OR AlignementRotaryTableWithPositioningsBlockHigh OR AlignementRotaryTableWithPositioningsBlockLow OR RotaryTableMotorBlockHigh OR Remove)
		(*Non appena viene espulsa la base che era presente nella stazione1 della giostra, ovvero quando la tavola � a 180� e la base � giunta alla slitta tra giostra e robot,
		la circonferenza che rappresenta la stazione1 nella vista dall'alto della tavola, dovr� essere resettata al colore BIANCO*)
THEN 	IF RotaryTablePosition>=180 AND RotaryTablePosition<=200 (*ho messo un range per essere sicuro*)
		THEN Color1:=ColorCircle[2];
				ElementOneTableCharged:=FALSE;
		END_IF;

		(*Non appena viene espulsa la base che era presente nella stazione2 della giostra, ovvero quando la tavola � a 240� e la base � giunta alla slitta tra giostra e robot,
		la circonferenza che rappresenta la stazione2 nella vista dall'alto della tavola, dovr� essere resettata al colore BIANCO*)
		IF RotaryTablePosition>=240 AND RotaryTablePosition<=260
		THEN Color2:=ColorCircle[2];
				ElementTwoTableCharged:=FALSE;
		END_IF;

		IF RotaryTablePosition>=300 AND RotaryTablePosition<=320
		THEN Color3:=ColorCircle[2];
				ElementThreeTableCharged:=FALSE;
		END_IF;

		IF RotaryTablePosition>=0 AND RotaryTablePosition<=20
		THEN Color4:=ColorCircle[2];
				ElementFourTableCharged:=FALSE;
		END_IF;

		IF RotaryTablePosition>=60 AND RotaryTablePosition<=80
		THEN Color5:=ColorCircle[2];
				ElementFiveTableCharged:=FALSE;
		END_IF;

		IF RotaryTablePosition>=120 AND RotaryTablePosition<=140
		THEN Color6:=ColorCircle[2];
				ElementSixTableCharged:=FALSE;
		END_IF;
END_IF;

(*Associo ad ogni stazione un'angolo : stazione1 ==>position=0, stazione2 ==> position=300,
stazione3 ==> position=240, stazione4 ==> position=180, stazione5 ==> position=120 e stazione6 ==> position=60*)

(*Ogni volta ke la tavola � allineata ed � disponibile un pezzo:*)

IF (RotaryTablePosition>=358 OR RotaryTablePosition<=2) AND ElementAirVsRotaryTable (*Quando la stazione1 della tavola � in corrispondenza della slitta tra verifica e lavorazione...*)
THEN	IF ElementAirRed (*Se dalla slitta sta arrivando una base rossa, *)
		THEN Color1:=ColorCircle[3];  (*allora faccio diventare rossa la circonferenza che rappresenta la stazione1 nella giostra vista dall'alto*)
				ElementOneTableRed:=TRUE; (*e attivo la variabile che corrisponde alla presenza di una base rossa nella stazione1 della tavola rotante*)
		ELSE ElementOneTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver (*Se dalla slitta arriva una base argento, *)
		THEN Color1:=ColorCircle[4];  (*allora faccio diventare argento la circonferenza che rappresenta la stazione1 nella giostra vista dall'alto*)
				ElementOneTableSilver:=TRUE;  (*e attivo la variabile che corrisponde alla presenza di una base argento nella stazione1 della tavola rotante*)
		ELSE ElementOneTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack  (*Se dalla slitta arriva una base nera, *)
		THEN Color1:=ColorCircle[1];    (*allora faccio diventare nera la circonferenza che rappresenta la stazione1 nella giostra vista dall'alto*)
				ElementOneTableBlack:=TRUE;  (*e attivo la variabile che corrisponde alla presenza di una base nera nella stazione1 della tavola rotante*)
		ELSE ElementOneTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned  (*Se la base che sta arrivando dalla slitta � Overturned, allora attivo la variabile che corrisponde ad una base Overturned nella stazione1*)
		THEN ElementOneTableOverturned:=TRUE;
		ELSE ElementOneTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack) (*Se non � in arrivo nessuna base, allora la circonferenza dovr� rimanere bianca*)
		THEN 	Color1:=ColorCircle[2];
				ElementOneTableCharged:=FALSE; (*e la variabile che indica la prsenza di una base nella stazione1 dovr� essere settata a false*)
		ELSE ElementOneTableCharged:=TRUE;
		END_IF;
END_IF;

IF (RotaryTablePosition>=58 AND RotaryTablePosition<=62) AND ElementAirVsRotaryTable
THEN	IF ElementAirRed
		THEN Color2:=ColorCircle[3];
				ElementTwoTableRed:=TRUE;
		ELSE ElementTwoTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver
		THEN Color2:=ColorCircle[4];
				ElementTwoTableSilver:=TRUE;
		ELSE ElementTwoTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack
		THEN Color2:=ColorCircle[1];
				ElementTwoTableBlack:=TRUE;
		ELSE ElementTwoTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned
		THEN ElementTwoTableOverturned:=TRUE;
		ELSE ElementTwoTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack)
		THEN 	Color2:=ColorCircle[2];
				ElementTwoTableCharged:=FALSE;
		ELSE ElementTwoTableCharged:=TRUE;
		END_IF;
END_IF;

IF (RotaryTablePosition>=118 AND RotaryTablePosition<=122) AND ElementAirVsRotaryTable
THEN	IF ElementAirRed
		THEN Color3:=ColorCircle[3];
				ElementThreeTableRed:=TRUE;
		ELSE ElementThreeTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver
		THEN Color3:=ColorCircle[4];
				ElementThreeTableSilver:=TRUE;
		ELSE ElementThreeTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack
		THEN Color3:=ColorCircle[1];
				ElementThreeTableBlack:=TRUE;
		ELSE ElementThreeTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned
		THEN ElementThreeTableOverturned:=TRUE;
		ELSE ElementThreeTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack)
		THEN 	Color3:=ColorCircle[2];
				ElementThreeTableCharged:=FALSE;
		ELSE ElementThreeTableCharged:=TRUE;
		END_IF;
END_IF;

IF (RotaryTablePosition>=178 AND RotaryTablePosition<=182) AND ElementAirVsRotaryTable
THEN	IF ElementAirRed
		THEN Color4:=ColorCircle[3];
				ElementFourTableRed:=TRUE;
		ELSE ElementFourTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver
		THEN Color4:=ColorCircle[4];
				ElementFourTableSilver:=TRUE;
		ELSE ElementFourTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack
		THEN Color4:=ColorCircle[1];
				ElementFourTableBlack:=TRUE;
		ELSE ElementFourTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned
		THEN ElementFourTableOverturned:=TRUE;
		ELSE ElementFourTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack)
		THEN 	Color4:=ColorCircle[2];
				ElementFourTableCharged:=FALSE;
		ELSE ElementFourTableCharged:=TRUE;
		END_IF;
END_IF;

IF (RotaryTablePosition>=238 AND RotaryTablePosition<=242) AND ElementAirVsRotaryTable
THEN	IF ElementAirRed
		THEN Color5:=ColorCircle[3];
				ElementFiveTableRed:=TRUE;
		ELSE ElementFiveTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver
		THEN Color5:=ColorCircle[4];
				ElementFiveTableSilver:=TRUE;
		ELSE ElementFiveTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack
		THEN Color5:=ColorCircle[1];
				ElementFiveTableBlack:=TRUE;
		ELSE ElementFiveTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned
		THEN ElementFiveTableOverturned:=TRUE;
		ELSE ElementFiveTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack)
		THEN 	Color5:=ColorCircle[2];
				ElementFiveTableCharged:=FALSE;
		ELSE ElementFiveTableCharged:=TRUE;
		END_IF;
END_IF;

IF (RotaryTablePosition>=298 AND RotaryTablePosition<=302) AND ElementAirVsRotaryTable
THEN	IF ElementAirRed
		THEN Color6:=ColorCircle[3];
				ElementSixTableRed:=TRUE;
		ELSE ElementSixTableRed:=FALSE;
		END_IF;
		IF ElementAirSilver
		THEN Color6:=ColorCircle[4];
				ElementSixTableSilver:=TRUE;
		ELSE ElementSixTableSilver:=FALSE;
		END_IF;
		IF ElementAirBlack
		THEN Color6:=ColorCircle[1];
				ElementSixTableBlack:=TRUE;
		ELSE ElementSixTableBlack:=FALSE;
		END_IF;
		IF ElementAirOverturned
		THEN ElementSixTableOverturned:=TRUE;
		ELSE ElementSixTableOverturned:=FALSE;
		END_IF;
		IF (NOT ElementAirRed AND NOT ElementAirSilver AND NOT ElementAirBlack)
		THEN 	Color6:=ColorCircle[2];
				ElementSixTableCharged:=FALSE;
		ELSE ElementSixTableCharged:=TRUE;
		END_IF;
END_IF;

(*Se l'elemento presente nella stazione1 della giostra, � Overturned, allora viene visualizzata la O all'interno della base1*)
IF ElementOneTableOverturned
THEN ElementOneTableO:='O';
ELSE	ElementOneTableO:=' ';
END_IF;

(*Se l'elemento presente nella stazione2 della giostra, � Overturned, allora viene visualizzata la O all'interno della base2*)
IF ElementTwoTableOverturned
THEN ElementTwoTableO:='O';
ELSE	ElementTwoTableO:=' ';
END_IF;

IF ElementThreeTableOverturned
THEN ElementThreeTableO:='O';
ELSE	ElementThreeTableO:=' ';
END_IF;

IF ElementFourTableOverturned
THEN ElementFourTableO:='O';
ELSE	ElementFourTableO:=' ';
END_IF;

IF ElementFiveTableOverturned
THEN ElementFiveTableO:='O';
ELSE	ElementFiveTableO:=' ';
END_IF;

IF ElementSixTableOverturned
THEN ElementSixTableO:='O';
ELSE	ElementSixTableO:=' ';
END_IF;

(*********************************************************************************************************************)


(*Simulazione del funzionamento del sensore di presenza nella stazione di foratura*)
IF RotaryTablePosition=60
THEN IF  (ElementOneTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementOneTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN 	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE 	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE 	IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=120
THEN IF  (ElementTwoTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementTwoTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN 	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE 	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE	IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=180
THEN IF  (ElementThreeTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementThreeTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN 	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE 	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE   IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=240
THEN IF  (ElementFourTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementFourTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE 	IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=300
THEN IF  (ElementFiveTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementFiveTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN 	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE 	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE 	IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=0
THEN IF (ElementSixTableCharged AND NOT AvailableLoadInControlPositioningBlockLow) OR AvailableLoadInControlPositioningBlockHigh
		THEN 	AvailableLoadInControlPositioning:=TRUE;
				IF InspectPosition=InspectDepth AND ((NOT ElementSixTableOverturned AND NOT InControlLoadInWrongPositionToBeDrilledBlockLow) OR InControlLoadInWrongPositionToBeDrilledBlockHigh)
				THEN 	InControlLoadInWrongPositionToBeDrilled:=TRUE;
				ELSE 	InControlLoadInWrongPositionToBeDrilled:=FALSE;
				END_IF;
		ELSE 	IF NOT AvailableLoadInControlPositioningBlockHigh
				THEN AvailableLoadInControlPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

(*Bloccaggio alto indipendente dalla posizione della tavola*)
IF FLAGInControlLoadInWrongPositionToBeDrilled (*settaggio del FLAG a righa 485 *)
THEN InControlLoadInWrongPositionToBeDrilled:=TRUE;
END_IF;

IF AvailableLoadInControlPositioningBlockHigh
THEN AvailableLoadInControlPositioning:=TRUE;
END_IF;

IF AvailableLoadInControlPositioningBlockLow
THEN AvailableLoadInControlPositioning:=FALSE;
END_IF;

(*FORATURA*)
(*---------------------------------------------------------------------------------------------------------------------------*)

(*Quando la tavola raggiunge i 120�, se � presente un pezzo nella stazione1, si attiva il sensore di presenza presente nella stazione di foratura *)
IF RotaryTablePosition=120
THEN IF (ElementOneTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE 	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;
(*Quando la tavola raggiunge i 180�, se � presente un pezzo nella stazione2, si attiva il sensore di presenza presente nella stazione di foratura *)
IF RotaryTablePosition=180
THEN IF (ElementTwoTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=240
THEN IF  (ElementThreeTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE 	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=300
THEN IF  (ElementFourTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=0
THEN IF  (ElementFiveTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE 	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF RotaryTablePosition=60
THEN IF (ElementSixTableCharged AND NOT AvailableLoadInDrillingPositioningBlockLow) OR AvailableLoadInDrillingPositioningBlockHigh
		THEN 	AvailableLoadInDrillingPositioning:=TRUE;
		ELSE 	IF NOT AvailableLoadInDrillingPositioningBlockHigh
				THEN AvailableLoadInDrillingPositioning:=FALSE;
				END_IF;
		END_IF;
END_IF;

IF  AvailableLoadInDrillingPositioningBlockHigh
THEN  AvailableLoadInDrillingPositioning:=TRUE;
END_IF;

IF  AvailableLoadInDrillingPositioningBlockLow
THEN  AvailableLoadInDrillingPositioning:=FALSE;
END_IF;

(*--------------------------------------------------------------------------------------------------------------------------------*)


(*STAZIONE DI CONTROLLO*)
(*------------------------------------------------------------------------------------------------------*)
(*Se l'elemento che si trova nella stazione di controllo � Overturned, allora la variabile ElementInControlOverturned:=TRUE*)
IF (RotaryTablePosition =60 AND ElementOneTableOverturned) OR (RotaryTablePosition =120 AND ElementTwoTableOverturned) OR (RotaryTablePosition =180 AND ElementThreeTableOverturned) OR (RotaryTablePosition =240 AND ElementFourTableOverturned) OR (RotaryTablePosition =300 AND ElementFiveTableOverturned) OR (RotaryTablePosition =0 AND ElementSixTableOverturned) AND NOT InControlLoadInWrongPositionToBeDrilledBlockHigh
THEN ElementInControlOverturned:=TRUE;
ELSE ElementInControlOverturned:=FALSE;
END_IF;
(*Se la base � Overturned, allora la profondit� a cui andr� l'utensile di controllo � minore rispetto al caso di una base corretta*)
IF ElementInControlOverturned
THEN InspectDepth:=9;
ELSE InspectDepth:=18;
END_IF;

(*Simulazione del tastatore*)
(*Se viene dato il comando di abbassare la torretta di controllo, allora la variabile associata alla posizione della torretta, viene incrementata, altrimenti essa viene decrementata*)
IF FLAGToLowerCylinderToInspectLoadBlockHigh OR (ToLowerCylinderToInspectLoad AND NOT ToLowerCylinderToInspectLoadBlockLow)
THEN InspectPosition:=InspectPosition+2;
		IF ToLowerCylinderToInspectLoadBlockHigh
		THEN FLAGToLowerCylinderToInspectLoadBlockHigh:=TRUE;
		ELSE FLAGToLowerCylinderToInspectLoadBlockHigh:=FALSE;
		END_IF;
ELSE  InspectPosition:=InspectPosition-2;
END_IF;

(*Quando l'utensile raggiunge la profondit� desiderata, si impone il non superamento di essa*)
IF InspectPosition>=InspectDepth
THEN	InspectPosition:=InspectDepth;
END_IF;

IF InspectPosition<=0
THEN InspectPosition:=0;
END_IF;
(*---------------------------------------------------------------------------------------*)


(*FORATURA*)
(*-----------------------------------------------------------------------------*)
(*Se il trapano � attivo, allora l'utensile in visualizzazione cambia colore*)
IF DrillingUnitActiveBlockHigh OR (DrillingUnitActive AND NOT DrillingUnitActiveBlockLow)
THEN DrillingToolColor:=TRUE;
ELSE DrillingToolColor:=FALSE;
END_IF;

(*Se viene attivato il pistone di bloccaggio del pezzo, nella stazione di foratura, allora viene incrementata la posizione di esso, finch� non raggiunge il fine corsa superiore*)
IF FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh OR (BlockingCylinderForwardInDrillingPositioning AND NOT BlockingCylinderForwardInDrillingPositioningBlockLow)
THEN 	IF BlockingActuator<24
		THEN BlockingActuator:=BlockingActuator+7;
		ELSE BlockingActuator:=24;
		END_IF;
		IF BlockingCylinderForwardInDrillingPositioningBlockHigh
		THEN FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh:=TRUE;
		ELSE FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh:=FALSE;
		END_IF;
(*Decemento della posizione del pistone, sino all'arrivo al fine corsa inferiore*)
ELSE 	IF BlockingActuator>0
		THEN BlockingActuator:=BlockingActuator-7;
		ELSE BlockingActuator:=0;
		END_IF;
END_IF;

(*Simulazione operazione di foratura*)

(*Se viene richiesto di abbassare la torretta di foratura, allora la variabile associata alla posizione della torretta viene incrementata*)
IF FLAGToLowerDrillingUnitBlockHigh OR (ToLowerDrillingUnit AND NOT ToLowerDrillingUnitBlockLow)
THEN DrillingUnitPosition:=DrillingUnitPosition+2;
		IF ToLowerDrillingUnitBlockHigh
		THEN FLAGToLowerDrillingUnitBlockHigh:=TRUE;
		ELSE FLAGToLowerDrillingUnitBlockHigh:=FALSE;
		END_IF;
END_IF;
(*Se viene richiesto di sollevare la torretta di foratura, allora la variabile di posizione viene decrementata*)
IF FLAGToLiftDrillingUnitBlockHigh OR (ToLiftDrillingUnit AND NOT ToLiftDrillingUnitBlockLow)
THEN DrillingUnitPosition:=DrillingUnitPosition-2;
		IF ToLiftDrillingUnitBlockHigh
		THEN FLAGToLiftDrillingUnitBlockHigh:=TRUE;
		ELSE FLAGToLiftDrillingUnitBlockHigh:=FALSE;
		END_IF;
END_IF;

(*Quando la torretta giunge al fine corsa inferiore, si attiva il sensore di presenza inferiore "DrillingUnitDown" *)
IF DrillingUnitPosition>=22
THEN 	DrillingUnitPosition:=22;
END_IF;

IF ToLowerDrillingUnit
THEN 	IF DrillingUnitPosition>=20
		THEN 	IF NOT DrillingUnitDownBlockLow
				THEN	DrillingUnitDown:=TRUE;
				ELSE 	DrillingUnitDown:=FALSE;
				END_IF;
		END_IF;
ELSE 	IF NOT DrillingUnitDownBlockHigh
		THEN DrillingUnitDown:=FALSE;
		END_IF;
END_IF;

(*Quando la torretta raggiunge il fine corsa superiore, si attiva il sensore di presenza superiore "DrillingUnitUp" *)
IF DrillingUnitPosition<=0
THEN 	DrillingUnitPosition:=0;
END_IF;

IF ToLiftDrillingUnit
THEN	IF DrillingUnitPosition<3
		THEN	IF NOT DrillingUnitUpBlockLow
				THEN DrillingUnitUp:=TRUE;
				ELSE DrillingUnitUp:=FALSE;
				END_IF;
		END_IF;
ELSE 	IF NOT DrillingUnitUpBlockHigh
		THEN DrillingUnitUp:=FALSE;
		END_IF;
END_IF;

IF DrillingUnitUpBlockHigh
THEN  DrillingUnitUp:=TRUE;
END_IF;
IF DrillingUnitDownBlockHigh
THEN  DrillingUnitDown:=TRUE;
END_IF;
IF DrillingUnitUpBlockHigh
THEN  DrillingUnitUp:=TRUE;
END_IF;
IF DrillingUnitDownBlockLow
THEN  DrillingUnitDown:=FALSE;
END_IF;

(*------------------------------------------------------------------------------------*)


(*INCREMENTO DELLE VARIABILI ASSOCIATE ALLA POSIZIONE LINEARE DELLE BASI SULLA GIOSTRA*)

IF RotaryTablePosition>0 AND RotaryTablePosition<=180 (*Tra 0-180� la base deve traslare orizzontalmente dalla posizione iniziale a quella di espulsione*)
THEN 	LinearTablePosition1:=RotaryTablePosition*1.4; (*LinearTablePosition1 � la variabile di posizione associata alla base1, ed � proporzionale alla posizione della tavola rotante, con un fattore di 1,4. Quest'ultimo � stato ricavato imponendo che la base raggiunga la posizione di espulsione al raggiungimento dei 180� *)
ELSE LinearTablePosition1:=0; (*Quando la base viene espulsa, essa diventa invisibile e quindi pu� essere riportata alla posizione iniziale*)
END_IF;

IF RotaryTablePosition>60 AND RotaryTablePosition<=240
THEN 	LinearTablePosition2:=(RotaryTablePosition-60)*1.4;
ELSE LinearTablePosition2:=0;
END_IF;

IF RotaryTablePosition>120 AND RotaryTablePosition<=300
THEN 	LinearTablePosition3:=(RotaryTablePosition-120)*1.4;
ELSE LinearTablePosition3:=0;
END_IF;

(*CASI PARTICOLARI DOVUTI AL FATTO CHE QUANDO RotaryTable=360 ==> RotaryTable:=0*)
(*--------------------------------------------------------------------------------------------------------------------------------------*)
IF RotaryTablePosition>180 AND (RotaryTablePosition<=360 OR RotaryTablePosition=0)
THEN 	LinearTablePosition4:=ABS(RotaryTablePosition-180)*1.4; (*ABS � il valore assoluto. Esso � necessario dal momento in cui RotaryTablePosition=0,  per ottenere una posizione lineare positiva*)
END_IF;

IF RotaryTablePosition>2 AND RotaryTablePosition<=180
THEN LinearTablePosition4:=0;
END_IF;

IF RotaryTablePosition>240 AND (RotaryTablePosition<=360 OR RotaryTablePosition=0)
THEN LinearTablePosition5:=ABS(RotaryTablePosition-240)*1.4;
END_IF;

IF (RotaryTablePosition>2 AND RotaryTablePosition<=60)
THEN LinearTablePosition5:=(RotaryTablePosition+120)*1.4;
END_IF;

IF (RotaryTablePosition>60 AND RotaryTablePosition<=240)
THEN LinearTablePosition5:=0;
END_IF;

IF RotaryTablePosition>300 AND (RotaryTablePosition<=360 OR RotaryTablePosition=0)
THEN LinearTablePosition6:=ABS(RotaryTablePosition-300)*1.4;
END_IF;

IF (RotaryTablePosition>2 AND RotaryTablePosition<=120)
THEN LinearTablePosition6:=(RotaryTablePosition+60)*1.4;
END_IF;

IF (RotaryTablePosition>120 AND RotaryTablePosition<=300)
THEN LinearTablePosition6:=0;
END_IF;
(*---------------------------------------------------------------------------------------------------------------------------------------*)

(*Stazione di Espulsione*)

(*Buffer per riportare l'informazione alla slitta prima del robot*)
IF RotaryTablePosition=180 (*Quando la tavola � a 180�, allora viene espulsa la base1*)
THEN 	ElementSleighRed:=ElementOneTableRed; (*BUFFER che trasferisce le informazioni della base che si trova nella stazione di espulsione, alla slitta tra giostra e robot *)
		ElementSleighBlack:=ElementOneTableBlack;
		ElementSleighSilver:=ElementOneTableSilver;
		ElementSleighOverturned:=ElementOneTableOverturned;
		IF ElementOneTableCharged (*Se � presente una base nella stazione di espulsione, allora viene attivato il sensore "virtuale" di presenza*)
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE; (*Sensore "virtuale, perch� non prensente veramente nell'impianto" settato a true*)
		END_IF;
END_IF;

IF RotaryTablePosition=240 (*Quando la tavola � a 240�, allora viene espulsa la base2*)
THEN 	ElementSleighRed:=ElementTwoTableRed;
		ElementSleighBlack:=ElementTwoTableBlack;
		ElementSleighSilver:=ElementTwoTableSilver;
		ElementSleighOverturned:=ElementTwoTableOverturned;
		IF ElementTwoTableCharged
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE;
		END_IF;
END_IF;

IF RotaryTablePosition=300
THEN 	ElementSleighRed:=ElementThreeTableRed;
		ElementSleighBlack:=ElementThreeTableBlack;
		ElementSleighSilver:=ElementThreeTableSilver;
		ElementSleighOverturned:=ElementThreeTableOverturned;
		IF ElementThreeTableCharged
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE;
		END_IF;
END_IF;

IF RotaryTablePosition=0
THEN 	ElementSleighRed:=ElementFourTableRed;
		ElementSleighBlack:=ElementFourTableBlack;
		ElementSleighSilver:=ElementFourTableSilver;
		ElementSleighOverturned:=ElementFourTableOverturned;
		IF ElementFourTableCharged
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE;
		END_IF;
END_IF;

IF RotaryTablePosition=60
THEN 	ElementSleighRed:=ElementFiveTableRed;
		ElementSleighBlack:=ElementFiveTableBlack;
		ElementSleighSilver:=ElementFiveTableSilver;
		ElementSleighOverturned:=ElementFiveTableOverturned;
		IF ElementFiveTableCharged
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE;
		END_IF;
END_IF;

IF RotaryTablePosition=120
THEN 	ElementSleighRed:=ElementSixTableRed;
		ElementSleighBlack:=ElementSixTableBlack;
		ElementSleighSilver:=ElementSixTableSilver;
		ElementSleighOverturned:=ElementSixTableOverturned;
		IF ElementSixTableCharged
		THEN 	AvailableLoadInExpulsionPositioning:=TRUE;
		END_IF;
END_IF;

(*Se l'elemento presente nella slitta di espulsione tra giostra e robot, � Overturned, allora viene visualizzata la O all'interno della base*)
IF ElementSleighOverturned
THEN ElementSleighO:='O';
ELSE	ElementSleighO:=' ';
END_IF;

(*Attivando la leva di espulsione, viene incrementata la sua posizione*)
IF FLAGExpellingLeverActiveBlockHigh OR (ExpellingLeverActive AND NOT ExpellingLeverActiveBlockLow)
THEN	LeverPosition:=LeverPosition+10;
		IF ExpellingLeverActiveBlockHigh
		THEN FLAGExpellingLeverActiveBlockHigh:=TRUE;
		ELSE FLAGExpellingLeverActiveBlockHigh:=FALSE;
		END_IF;
ELSE	LeverPosition:=LeverPosition-10;
END_IF;

IF LeverPosition<0
THEN	LeverPosition:=0;
END_IF;

(*Se il cilindro di espulsione raggiunge il fine corsa, il pezzo e' pronto per l'assemblaggio*)
IF LeverPosition>=50
THEN	LeverPosition:=50;
		ElementSleighCharged:=TRUE;
END_IF;

(*Quando � presente un pezzo sulla slitta, viene incrementata la sua posizione*)
IF ElementSleighCharged
THEN MovementElementSleigh:=MovementElementSleigh+2;
ELSE MovementElementSleigh:=0;
END_IF;

(*Quando la base raggiunge la fine della slitta, viene resa invisibile, passando le sue informazioni e rendendo visibile la base nella "Stazione1" (da qui indicher� con Stazione1 la stazione dopo la giostra)*)
IF (MovementElementSleigh>=10)
THEN	MovementElementSleigh:=10;
		ElementSleighCharged:=FALSE;
		ElementStation1RobotCharged:=TRUE;
		ElementStation1RobotRed:=ElementSleighRed;
		ElementStation1RobotBlack:=ElementSleighBlack;
		ElementStation1RobotSilver:=ElementSleighSilver;
		ElementStation1RobotOverturned:=ElementSleighOverturned;
		IF NOT AvailableLoadForRobotBlockLow
		THEN AvailableLoadForRobot:=TRUE; (*si attiva il sensore di presenza pezzo nella stazione tra giostra e robot*)
		END_IF;
END_IF;

IF AvailableLoadForRobotBlockHigh
THEN AvailableLoadForRobot:=TRUE;
END_IF;

(*Se la base presente nella Stazione1 � Overturned, rendo visibile la O all'interno di essa.*)
IF ElementStation1RobotOverturned
THEN ElementStation1RobotO:='O';
ELSE ElementStation1RobotO:='';
END_IF;


IF Remove
THEN	Color1:=ColorCircle[2];
		Color2:=ColorCircle[2];
		Color3:=ColorCircle[2];
		Color4:=ColorCircle[2];
		Color5:=ColorCircle[2];
		Color6:=ColorCircle[2];
		ElementAirVsRotaryTable:=FALSE;
		ElementMeasureCharged:=FALSE;
		AlignementRotaryTableWithPositionings:=FALSE;
		AvailableLoadForWorkingStation:=FALSE;
		AvailableLoadInControlPositioning:=FALSE;
		AvailableLoadInDrillingPositioning:=FALSE;
		AvailableLoadInExpulsionPositioning:=FALSE;
		InControlLoadInWrongPositionToBeDrilled:=FALSE;
		LeverPosition:=0;
		ElementSleighRed:=FALSE;
		ElementSleighBlack:=FALSE;
		ElementSleighSilver:=FALSE;
		ElementSleighCharged:=FALSE;
		ElementSleighOverturned:=FALSE;
		ElementOneTableCharged:=FALSE;
		ElementTwoTableCharged:=FALSE;
		ElementThreeTableCharged:=FALSE;
		ElementFourTableCharged:=FALSE;
		ElementFiveTableCharged:=FALSE;
		ElementSixTableCharged:=FALSE;
END_IF;

IF FaultDetected
THEN 	FLAGRotaryTableMotorBlockHigh:=FALSE;
		FLAGToLowerCylinderToInspectLoadBlockHigh:=FALSE;
		FLAGBlockingCylinderForwardInDrillingPositioningBlockHigh:=FALSE;
		FLAGToLowerDrillingUnitBlockHigh:=FALSE;
		FLAGToLiftDrillingUnitBlockHigh:=FALSE;
		FLAGExpellingLeverActiveBlockHigh:=FALSE;
END_IF;               2   , / / ��           PlantMagazzino ��d	��d      ��������        �   PROGRAM PlantMagazzino
VAR
	BlockedVsVerification: BOOL; (*flag che segnala il bloccaggio del braccio rotante in posizione verifica*)
	BlockedVsWarehouse: BOOL; (*flag che segnala il bloccaggio del braccio rotante in posizione magazzino*)
END_VAR�.  (*sensore di presenza pezzo magazzino: *)
IF NOT (EmptyWarehouseBlockHigh OR EmptyWarehouseBlockLow) THEN (*sensore non guasto*)
	IF NOT ElementOneCharged AND NOT ElementTwoCharged THEN (*se non ho elementi caricati in magazzino attivo il sensore*)
			EmptyWarehouse:=TRUE;
		ELSE EmptyWarehouse:=FALSE;
	END_IF;
END_IF;

(*sensore di presenza pezzo magazzino guasto: *)
IF EmptyWarehouseBlockHigh THEN (*bloccato alto*)
	EmptyWarehouse:=TRUE;
END_IF;
IF EmptyWarehouseBlockLow THEN (*bloccato basso*)
	EmptyWarehouse:=FALSE;
END_IF;

(*se il comando di uscita del cilindro e' attivo, e il cilindro non � bloccato, incremento la posizione dello stesso*)
IF NOT (CylinderExtractsLoadFromWarehouseBlock) THEN
	IF CylinderExtractsLoadFromWarehouse AND CylinderPosition<=80 THEN
			CylinderPosition:=CylinderPosition+4;
			ElementPosition:=ElementPosition+4;
		ELSE IF NOT CylinderExtractsLoadFromWarehouse AND CylinderPosition>=0 AND NOT CylinderExtractsLoadFromWarehouseBlockHigh  THEN
				CylinderPosition:=CylinderPosition-4; (*altrimenti, se l'attuatore non � bloccato alto, torna in posizione retratta grazie all'inversione del flusso d'aria*)
				ElementPosition:=0;
			END_IF;
	END_IF;
END_IF;

IF Reset THEN
	CylinderPosition:=0;
END_IF;

(*sensore di finecorsa inferiore cilindro di estrazione: *)
IF (CylinderPosition<1) THEN  (*Controllo che permette al cilindro di mantenersi in posizione di riposo quando non e' attivo il comando di attuazione*)
	CylinderPosition:=0;
	IF NOT (CylinderExtractionLoadInRetroactivePositionBlockHigh OR CylinderExtractionLoadInRetroactivePositionBlockLow) THEN
		CylinderExtractionLoadInRetroactivePosition:=TRUE; (*Sensore attivo quando il cilindro e' in posizione retratta e sensore non gausto*)
	END_IF;
	CylinderBehind:=FALSE; (*Flag vero solo quando il cilindro sta tornardo in posizione di riposo*)
	ELSE IF NOT (CylinderExtractionLoadInRetroactivePositionBlockHigh OR CylinderExtractionLoadInRetroactivePositionBlockLow) THEN
			CylinderExtractionLoadInRetroactivePosition:=FALSE;
		END_IF;
END_IF;

(*sensore di finecorsa inferiore cilindro di estrazione guasto: *)
IF CylinderExtractionLoadInRetroactivePositionBlockHigh THEN
	CylinderExtractionLoadInRetroactivePosition:=TRUE;
END_IF;
IF CylinderExtractionLoadInRetroactivePositionBlockLow THEN
	CylinderExtractionLoadInRetroactivePosition:=FALSE;
END_IF;

IF ElementPosition>=80  THEN
		ElementPosition:=80;
		ReadyForRotaryMaker:=TRUE; (*Quando si trova in posizione "80", il pezzo e' pronto per essere caricato dal RotaryMaker*)
	ELSE
		ReadyForRotaryMaker:=FALSE;
END_IF;

(*sensore di finecorsa superiore cilindro di estrazione: *)
IF CylinderPosition>=80  THEN
	CylinderPosition:=80;
	IF NOT (CylinderExtractionLoadInExtensivePositionBlockHigh OR CylinderExtractionLoadInExtensivePositionBlockLow) THEN
		CylinderExtractionLoadInExtensivePosition:=TRUE; (*Sensore attivo quando il cilindro e' in posizione totalmente estratta e sensore non guasto *)
	END_IF;
	ELSE IF NOT (CylinderExtractionLoadInExtensivePositionBlockHigh OR CylinderExtractionLoadInExtensivePositionBlockLow) THEN
			CylinderExtractionLoadInExtensivePosition:=FALSE;
		END_IF;
END_IF;

(*sensore di finecorsa superiore guasto: *)
IF CylinderExtractionLoadInExtensivePositionBlockHigh THEN
	CylinderExtractionLoadInExtensivePosition:=TRUE;
END_IF;
IF CylinderExtractionLoadInExtensivePositionBlockLow THEN
	CylinderExtractionLoadInExtensivePosition:=FALSE;
END_IF;

IF (CylinderPosition>18) AND NOT CylinderExtractsLoadFromWarehouse AND NOT CylinderExtractsLoadFromWarehouseBlockHigh THEN
CylinderBehind:=TRUE; (*Il flag "CylinderBehind" e' attivo quando il cilindro sta tornardo indietro verso la posizione di riposo*)
END_IF;

(*generatore di vuoto*)
IF NOT VacuumGeneratorBlock THEN
	IF VacuumGenerator AND NOT ExpulsionAirVacuumVis THEN
		VacuumGeneratorSim:=TRUE; (*Simulo il funzionamento del generatore di vuoto solo se non � attivo anche il soffio di espulsione*)
		ELSE IF NOT VacuumGeneratorBlockHigh THEN
				VacuumGeneratorSim:=FALSE;
			END_IF;
	END_IF;
END_IF;

(*sensore di vuoto*)
IF VacuumGeneratorSim THEN
		IF NOT (VacuumGeneratorOkBlockHigh OR VacuumGeneratorOkBlockLow OR NOT ElementRotaryCharged) THEN
			VacuumGeneratorOk:=TRUE;
		END_IF;
	ELSE IF NOT (VacuumGeneratorOkBlockHigh OR VacuumGeneratorOkBlockLow OR NOT ElementRotaryCharged) THEN
				VacuumGeneratorOk:=FALSE;
		END_IF;
END_IF;

IF NOT ElementRotaryCharged THEN (*se non ho un pezzo sotto alla ventosa non si genera il vuoto*)
	VacuumGeneratorOk:=FALSE;
END_IF;

(*soffio di rilascio pezzo*)
IF NOT ExpulsionAirVacuumBlock THEN
	IF NOT VacuumGeneratorBlockHigh AND ExpulsionAirVacuum THEN (*se ho il pezzo sull'ascensore avvio il rilascio*)
		ExpulsionAirVacuumVis:=TRUE;
		ELSE IF NOT ExpulsionAirVacuumBlockHigh THEN
				ExpulsionAirVacuumVis:=FALSE;
			END_IF;
	END_IF;
END_IF;

(*sensore generazione di vuoto guasto: *)
IF VacuumGeneratorOkBlockHigh THEN
	VacuumGeneratorOk:=TRUE;
END_IF;
IF VacuumGeneratorOkBlockLow THEN
	VacuumGeneratorOk:=FALSE;
END_IF;

(*braccio rotante*)
IF NOT RotaryMakerVsVerificationBlock THEN
	IF (RotaryMakerVsVerification OR BlockedVsVerification)  AND RotaryPosition<=180 AND NOT BlockedVsWarehouse  THEN  (*Il braccio va verso la stazione di verifica quando non � gi� a fine corsa e il relativo comando e' attivo, oppure � bloccato alto*)
		RotaryPosition:=RotaryPosition+9;
		IF RotaryMakerVsVerificationBlockHigh THEN  (*simulo il bloccaggio del braccio in posizione di verifica*)
			BlockedVsVerification:=TRUE;
			ELSE BlockedVsVerification:=FALSE;
		END_IF;
	END_IF;
END_IF;

IF NOT RotaryMakerVsWarehouseBlock THEN
	IF (RotaryMakerVsWarehouse OR BlockedVsWarehouse) AND RotaryPosition>=0 AND NOT BlockedVsVerification THEN  (*Il braccio va verso il magazzino quando non � gi� a fine corsa e il relativo comando e' attivo, oppure � bloccato alto*)
		RotaryPosition:=RotaryPosition-9;
		IF RotaryMakerVsWarehouseBlockHigh THEN  (*simulo il bloccaggio del braccio in posizione magazzino*)
			BlockedVsWarehouse:=TRUE;
			ELSE BlockedVsWarehouse:=FALSE;
		END_IF;
	END_IF;
END_IF;

IF RotaryPosition<1 THEN  (*Il braccio giunge a fine corsa verso il magazzino ed il relativo sensore si attiva*)
	RotaryPosition:=0;
	IF NOT (RotaryMakerInPositionWarehouseBlockHigh OR RotaryMakerInPositionWarehouseBlockLow) THEN
		RotaryMakerInPositionWarehouse:=TRUE; (*Questo sensore e' attivo quando il cilindro e' in posizione retratta*)
	END_IF;
	ELSE IF NOT (RotaryMakerInPositionWarehouseBlockHigh OR RotaryMakerInPositionWarehouseBlockLow) THEN
			RotaryMakerInPositionWarehouse:=FALSE;
		END_IF;
END_IF;

(*sensore finecorsa inferiore braccio rotante guasto*)
IF RotaryMakerInPositionWarehouseBlockHigh THEN
	RotaryMakerInPositionWarehouse:=TRUE;
END_IF;
IF RotaryMakerInPositionWarehouseBlockLow THEN
	RotaryMakerInPositionWarehouse:=FALSE;
END_IF;

IF RotaryPosition>=180 THEN (*Il braccio giunge a fine corsa verso la verifica ed il relativo sensore si attiva*)
	RotaryPosition:=180;
	IF NOT (RotaryMakerInPositionVerificationBlockHigh OR RotaryMakerInPositionVerificationBlockLow) THEN
		RotaryMakerInPositionVerification:=TRUE; (*sensore*)
	END_IF;
	ELSE IF NOT (RotaryMakerInPositionVerificationBlockHigh OR RotaryMakerInPositionVerificationBlockLow) THEN
			RotaryMakerInPositionVerification:=FALSE;
		END_IF;
END_IF;

(*sensore finecorsa superiore braccio rotante guasto*)
IF RotaryMakerInPositionVerificationBlockHigh THEN
	RotaryMakerInPositionVerification:=TRUE;
END_IF;
IF RotaryMakerInPositionVerificationBlockLow THEN
	RotaryMakerInPositionVerification:=FALSE;
END_IF;

(*attivazione della fotocellula per l'interferenza braccioRotante-ascensore*)
IF NOT (VerificationBusyBlockHigh OR VerificationBusyBlockLow) THEN
	IF RotaryPosition>140  OR LiftPosition>10 THEN
		VerificationBusy:=TRUE;
		ELSE
			VerificationBusy:=FALSE;
	END_IF;
END_IF;

(*fotocellula interferenza guasta: *)
IF VerificationBusyBlockHigh THEN
	VerificationBusy:=TRUE;
END_IF;
IF VerificationBusyBlockLow THEN
	VerificationBusy:=FALSE;
END_IF;

(*Se gli elementi sono capovolti, nella simulazione grafica questo viene segnalato da una "O" scritta sul pezzo*)
IF ElementOneOverturned THEN
	ElementOneO:='O';
	ELSE  IF NOT ElementOneShort THEN
		ElementOneO:='' ;
	END_IF;
END_IF;

IF ElementTwoOverturned THEN
	ElementTwoO:='O';
	ELSE  IF NOT ElementTwoShort THEN
		ElementTwoO:='' ;
	END_IF;
END_IF;

IF ElementThreeOverturned THEN
	ElementThreeO:='O';
	ELSE  IF NOT ElementThreeShort THEN
		ElementThreeO:='' ;
	END_IF;
END_IF;

IF ElementFourOverturned THEN
	ElementFourO:='O';
	ELSE  IF NOT ElementFourShort THEN
		ElementFourO:='' ;
	END_IF;
END_IF;

IF ElementFiveOverturned THEN
	ElementFiveO:='O';
	ELSE  IF NOT ElementFiveShort THEN
		ElementFiveO:='' ;
	END_IF;
END_IF;

IF ElementSixOverturned THEN
	ElementSixO:='O';
	ELSE  IF NOT ElementSixShort THEN
		ElementSixO:='' ;
	END_IF;
END_IF;

IF ElementSevenOverturned THEN
	ElementSevenO:='O';
	ELSE  IF NOT ElementSevenShort THEN
		ElementSevenO:='' ;
	END_IF;
END_IF;

IF ElementEightOverturned THEN
	ElementEightO:='O';
	ELSE  IF NOT ElementEightShort THEN
		ElementEightO:='' ;
	END_IF;
END_IF;

IF ElementWaitingOverturned THEN
	ElementWaitingO:='O';
	ELSE  IF NOT ElementWaitingShort THEN
		ElementWaitingO:='' ;
	END_IF;
END_IF;

IF ElementRotaryOverturned THEN
	ElementRotaryO:='O';
	ELSE  IF NOT ElementRotaryShort THEN
		ElementRotaryO:='' ;
	END_IF;
END_IF;

IF ElementVerificationOverturned THEN
	ElementVerificationO:='O';
	ELSE  IF NOT ElementVerificationShort THEN
		ElementVerificationO:='' ;
	END_IF;
END_IF;

IF ElementMeasureOverturned THEN
	ElementMeasureO:='O';
	ELSE  IF NOT ElementMeasureShort THEN
		ElementMeasureO:='' ;
	END_IF;
END_IF;

IF ElementAirOverturned THEN (*se c'era un pezzo corto ora � gi� stato scartato quindi nn lo considero pi�*)
	ElementAirO:='O'; ELSE
	ElementAirO:='' ;
END_IF;

IF ElementOneTableOverturned THEN
	ElementOneTableO:='O'; ELSE
	ElementOneTableO:='' ;
END_IF;

(*Se gli elementi sono corti, nella simulazione grafica questo viene segnalato da una "S" scritta sul pezzo*)
IF ElementOneShort THEN
	ElementOneO:='S';
	ELSE IF NOT ElementOneOverturned THEN
		ElementOneO:='' ;
	END_IF;
END_IF;

IF ElementTwoShort THEN
	ElementTwoO:='S';
	ELSE IF NOT ElementTwoOverturned THEN
		ElementTwoO:='' ;
	END_IF;
END_IF;

IF ElementThreeShort THEN
	ElementThreeO:='S';
	ELSE IF NOT ElementThreeOverturned THEN
		ElementThreeO:='' ;
	END_IF;
END_IF;

IF ElementFourShort THEN
	ElementFourO:='S';
	ELSE IF NOT ElementFourOverturned THEN
		ElementFourO:='' ;
	END_IF;
END_IF;

IF ElementFiveShort THEN
	ElementFiveO:='S';
	ELSE IF NOT ElementFiveOverturned THEN
		ElementFiveO:='' ;
	END_IF;
END_IF;

IF ElementSixShort THEN
	ElementSixO:='S';
	ELSE IF NOT ElementSixOverturned THEN
		ElementSixO:='' ;
	END_IF;
END_IF;

IF ElementSevenShort THEN
	ElementSevenO:='S';
	ELSE IF NOT ElementSevenOverturned THEN
		ElementSevenO:='' ;
	END_IF;
END_IF;

IF ElementEightShort THEN
	ElementEightO:='S';
	ELSE IF NOT ElementEightOverturned THEN
		ElementEightO:='' ;
	END_IF;
END_IF;

IF ElementWaitingShort THEN
	ElementWaitingO:='S';
	ELSE IF NOT ElementWaitingOverturned THEN
		ElementWaitingO:='' ;
	END_IF;
END_IF;


IF ElementRotaryShort THEN
	ElementRotaryO:='S';
	ELSE IF NOT ElementRotaryOverturned THEN
		ElementRotaryO:='' ;
	END_IF;
END_IF;

IF ElementVerificationShort THEN
	ElementVerificationO:='S';
	ELSE IF NOT ElementVerificationOverturned THEN
		ElementVerificationO:='' ;
	END_IF;
END_IF;

IF ElementMeasureShort THEN
	ElementMeasureO:='S';
	ELSE IF NOT ElementMeasureOverturned THEN
		ElementMeasureO:='' ;
	END_IF;
END_IF;               3   , ] ] �"           PlantScarico ��d	��d      ��������        "   PROGRAM PlantScarico
VAR
END_VAR  IF (CylinderBehind AND CylinderPosition<=4) THEN	(*Quando il cilindro di espulsione sta tornando in posizione di riposo, tutti i pezzi ricevono gli attributi dal pezzo che gli e' sopra*)
	ElementOneCharged:=ElementTwoCharged;
	ElementOneRed:=ElementTwoRed;
	ElementOneSilver:=ElementTwoSilver;
	ElementOneBlack:=ElementTwoBlack;
	ElementOneOverturned:=ElementTwoOverturned;
	ElementOneShort:=ElementTwoShort;

	ElementTwoCharged:=ElementThreeCharged;
	ElementTwoRed:=ElementThreeRed;
	ElementTwoSilver:=ElementThreeSilver;
	ElementTwoBlack:=ElementThreeBlack;
	ElementTwoOverturned:=ElementThreeOverturned;
	ElementTwoShort:=ElementThreeShort;

	ElementThreeCharged:=ElementFourCharged;
	ElementThreeRed:=ElementFourRed;
	ElementThreeSilver:=ElementFourSilver;
	ElementThreeBlack:=ElementFourBlack;
	ElementThreeOverturned:=ElementFourOverturned;
	ElementThreeShort:=ElementFourShort;

	ElementFourCharged:=ElementFiveCharged;
	ElementFourRed:=ElementFiveRed;
	ElementFourSilver:=ElementFiveSilver;
	ElementFourBlack:=ElementFiveBlack;
	ElementFourOverturned:=ElementFiveOverturned;
	ElementFourShort:=ElementFiveShort;

	ElementFiveCharged:=ElementSixCharged;
	ElementFiveRed:=ElementSixRed;
	ElementFiveSilver:=ElementSixSilver;
	ElementFiveBlack:=ElementSixBlack;
	ElementFiveOverturned:=ElementSixOverturned;
	ElementFiveShort:=ElementSixShort;

	ElementSixCharged:=ElementSevenCharged;
	ElementSixRed:=ElementSevenRed;
	ElementSixSilver:=ElementSevenSilver;
	ElementSixBlack:=ElementSevenBlack;
	ElementSixOverturned:=ElementSevenOverturned;
	ElementSixShort:=ElementSevenShort;

	ElementSevenCharged:=ElementEightCharged;
	ElementSevenRed:=ElementEightRed;
	ElementSevenSilver:=ElementEightSilver;
	ElementSevenBlack:=ElementEightBlack;
	ElementSevenOverturned:=ElementEightOverturned;
	ElementSevenShort:=ElementEightShort;

 	ElementEightCharged:=FALSE;
	ElementEightRed:=FALSE;
	ElementEightSilver:=FALSE;
	ElementEightBlack:=FALSE;
	ElementEightOverturned:=FALSE;
	ElementEightShort:=FALSE;

END_IF;               5   , C X �           PlantVerification ��d	��d      ��������        S  PROGRAM PlantVerification
VAR
	NoElement: BOOL; (*variabile che segnala se il pezzo � gia stato preso dal rotary maker (necessaria nel caso di estrattore-basi bloccato alto) *)
	BlockedHigh: BOOL; (*flag che segnala il bloccaggio alto dell'ascensore*)
	BlockedLow: BOOL; (*flag che segnala il bloccaggio basso dell'ascensore*)
END_VAR;5  (* Valore iniziale braccio rotante*)
IF FLAGRotaryPosition
THEN 	FLAGRotaryPosition:=FALSE;
		RotaryPosition:=RotaryPositionInitialVis+RotaryPosition;
END_IF;

(* Valore iniziale ascensore*)
IF FLAGLiftPosition
THEN 	FLAGLiftPosition:=FALSE;
		LiftPosition:=LiftPositionInitialVis+LiftPosition;
END_IF;

(*elemento in attesa di essere caricato dal RotaryMaker*)
IF CylinderExtractionLoadInExtensivePosition AND ElementOneCharged AND NOT CylinderExtractionLoadInExtensivePositionBlockHigh THEN	(*Quando il cilindro di espulsione dal magazzino e' arrivato a fine corsa, viene reso visibile il pezzo in posizione di attesa*)
	ElementWaitingCharged:=TRUE;
	ElementWaitingRed:=ElementOneRed;
	ElementWaitingBlack:=ElementOneBlack;
	ElementWaitingSilver:=ElementOneSilver;
	ElementWaitingOverturned:=ElementOneOverturned;
	ElementWaitingShort:=ElementOneShort;
	ElementOneCharged:=FALSE;
	NoElement:=TRUE;
END_IF;

IF CylinderBehind THEN
	NoElement:=FALSE;
END_IF;

(*elemento caricato dal braccio rotante*)
IF ElementWaitingCharged  AND RotaryMakerInPositionWarehouse THEN
	ElementRotaryCharged:=TRUE;
	ElementRotaryRed:=ElementWaitingRed;
	ElementRotaryBlack:=ElementWaitingBlack;
	ElementRotarySilver:=ElementWaitingSilver;
	ElementRotaryOverturned:=ElementWaitingOverturned;
	ElementRotaryShort:=ElementWaitingShort;
	ElementWaitingCharged:=FALSE;
END_IF;

(*----------------Plant della stazione di verifica---------------*)
(*ascensore*)
IF NOT  ToLiftCylinderToMeasureLoadBlock THEN
	IF (ToLiftCylinderToMeasureLoad OR BlockedHigh) AND LiftPosition<=140 AND NOT BlockedLow THEN (*l'ascensore sale quando Il relativo comando e' attivo o � bloccato alto*)
		LiftPosition:=LiftPosition+7;
		IF ToLiftCylinderToMeasureLoadBlockHigh THEN  (*simulo il bloccaggio dell'ascensore in alto*)
			BlockedHigh:=TRUE;
			ELSE BlockedHigh:=FALSE;
		END_IF;
	END_IF;
END_IF;

IF NOT  ToLowerCylinderToMeasureLoadBlock THEN
	IF (ToLowerCylinderToMeasureLoad OR BlockedLow) AND LiftPosition>=0 AND NOT BlockedHigh THEN (*l'ascensore scende quando Il relativo comando e' attivo o � bloccato alto*)
		LiftPosition:=LiftPosition-7;
		IF ToLowerCylinderToMeasureLoadBlockHigh THEN  (*simulo il bloccaggio dell'ascensore in basso*)
			BlockedLow:=TRUE;
			ELSE BlockedLow:=FALSE;
		END_IF;
	END_IF;
END_IF;

IF LiftPosition<1 THEN
	LiftPosition:=0;
END_IF;

IF LiftPosition>=140 THEN
	LiftPosition:=140;
END_IF;

(*sensore di finecorsa inferiore ascensore*)
IF NOT (CylinderDownToMeasureLoadBlockHigh OR CylinderDownToMeasureLoadBlockLow) THEN
	IF LiftPosition=0(*(LiftPosition>0 AND LiftPosition<=15) OR (LiftPosition>=30 AND LiftPosition<=45) *)THEN   (*Sensore di finecorsa inferiore dell'ascensore*)
			CylinderDownToMeasureLoad:=TRUE;
		ELSE CylinderDownToMeasureLoad:=FALSE;
	END_IF;
END_IF;

(*sensore di finecorsa inferiore ascensore guasto*)
IF CylinderDownToMeasureLoadBlockHigh THEN
	CylinderDownToMeasureLoad:=TRUE;
END_IF;
IF CylinderDownToMeasureLoadBlockLow THEN
	CylinderDownToMeasureLoad:=FALSE;
END_IF;

(*sensore di finecorsa superiore ascensore*)
IF NOT (CylinderUpToMeasureLoadBlockHigh OR CylinderUpToMeasureLoadBlockLow) THEN
	IF LiftPosition=140 THEN
			CylinderUpToMeasureLoad:=TRUE;
		ELSE CylinderUpToMeasureLoad:=FALSE;
	END_IF;
END_IF;

(*sensore di finecorsa superiore ascensore guasto*)
IF CylinderUpToMeasureLoadBlockHigh THEN
	CylinderUpToMeasureLoad:=TRUE;
END_IF;
IF CylinderUpToMeasureLoadBlockLow THEN
	CylinderUpToMeasureLoad:=FALSE;
END_IF;

(*cilindro di espulsione sull'ascensore*)
IF NOT ToExtendCylinderOfExtractionVsGuideBlock THEN
	IF ToExtendCylinderOfExtractionVsGuide AND CylinderOfExtractionPosition<=80 THEN  (*Quando il relativo comando e' attivo, il cilindro di espulsione sull'ascensore e' attivo, altrimenti torna a fine corsa*)
			CylinderOfExtractionPosition:=CylinderOfExtractionPosition+4;
		ELSE IF (NOT ToExtendCylinderOfExtractionVsGuide  AND CylinderOfExtractionPosition>=0  OR Fault) AND NOT ToExtendCylinderOfExtractionVsGuideBlockHigh THEN
				CylinderOfExtractionPosition:=CylinderOfExtractionPosition-4;
			END_IF;
	END_IF;
END_IF;

(*sensore di finecorsa cilindro di espulsione (solo inferiore) *)
IF CylinderOfExtractionPosition<1 THEN	(*Simulazione del cilindro di espulsione scarti*)
	CylinderOfExtractionPosition:=0;
	IF NOT (CylinderOfExtractionInRetroactivePositionBlockHigh OR CylinderOfExtractionInRetroactivePositionBlockLow) THEN
		CylinderOfExtractionInRetroactivePosition:=TRUE;
	END_IF;
	ELSE IF NOT (CylinderOfExtractionInRetroactivePositionBlockHigh OR CylinderOfExtractionInRetroactivePositionBlockLow) THEN
			CylinderOfExtractionInRetroactivePosition:=FALSE;
		END_IF;
END_IF;

(*sensore di finecorsa cilindro di espulsione (solo inferiore) guasto *)
IF CylinderOfExtractionInRetroactivePositionBlockHigh THEN
	CylinderOfExtractionInRetroactivePosition:=TRUE;
END_IF;
IF CylinderOfExtractionInRetroactivePositionBlockLow THEN
	CylinderOfExtractionInRetroactivePosition:=FALSE;
END_IF;

IF CylinderOfExtractionPosition>=80 THEN
	CylinderOfExtractionPosition:=80;
END_IF;

IF CylinderOfExtractionPosition>=75 THEN
	ElementVerificationCharged:=FALSE; (*Se ho espulso il pezzo non ho pi� l'elemento in verifica*)
END_IF;

(*verifica colore*)
IF NOT (ColourMeasurementBlockHigh OR ColourMeasurementBlockLow) THEN
	IF ElementVerificationBlack  AND CylinderOfExtractionPosition=0 THEN	(*Se l'elemento e' nero, il sensore di rilevamento colore non si accende*)
		ColourMeasurement:=FALSE;
	END_IF;
	IF (ElementVerificationSilver OR ElementVerificationRed)  AND CylinderOfExtractionPosition<70 AND (ElementVerificationCharged OR ElementMeasureCharged) THEN
			ColourMeasurement:=TRUE;
		ELSE ColourMeasurement:=FALSE;
	END_IF;
END_IF;

(*sensore di verifica colore guasto*)
IF ColourMeasurementBlockHigh THEN
	ColourMeasurement:=TRUE;
END_IF;
IF ColourMeasurementBlockLow THEN
	ColourMeasurement:=FALSE;
END_IF;

IF MisuratorPosition>=40 THEN
	MisuratorPosition:=40;
END_IF;

IF MisuratorPosition<1 THEN
	MisuratorPosition:=0;
END_IF;

IF (LiftPosition>135 AND MisuratorPosition>=0 AND (ElementVerificationCharged OR ElementMeasureCharged)) AND CylinderOfExtractionPosition<50  THEN
			MisuratorPosition:=MisuratorPosition-12;
END_IF;

(*uscita misuratore*)
IF NOT (MeasurementNotOkBlockHigh OR MeasurementNotOkBlockLow) THEN
	IF ElementMeasureBlack THEN	(*Se l'elemento � nero (cio� basso), la misura d� in uscita FALSE*)
		MeasurementNotOk:=FALSE;
	END_IF;
	IF (ElementMeasureSilver OR ElementMeasureRed) AND NOT ElementMeasureShort AND CylinderOfExtractionPosition<70 AND ElementMeasureCharged THEN (*Se l'elemento � chiaro normale (cio� alto), la misura d� in uscita TRUE*)
				MeasurementNotOk:=TRUE;
			ELSE MeasurementNotOk:=FALSE;
		END_IF;
	IF ElementMeasureShort THEN (*Se l'elemento � chiaro corto, la misura d� in uscita FALSE*)
		MeasurementNotOk:=FALSE;
		IF LiftPosition>2 THEN
				ElementVerificationCharged:=TRUE;  (*rivisualizzo l'elemento corto in Verification e lo nascondo da Measure*)
				ElementMeasureCharged:=FALSE;
				ElementVerificationShort:=ElementMeasureShort;
		END_IF;
	END_IF;
END_IF;

IF NOT ElementVerificationCharged THEN  (*se ho espulso il pezzo corto resetto le variabili che lo identificavano*)
		ElementVerificationShort:=FALSE;
		ElementMeasureShort:=FALSE;
END_IF;

(*sensore di misura rotto*)
IF MeasurementNotOkBlockHigh THEN
	MeasurementNotOk:=TRUE;
END_IF;
IF MeasurementNotOkBlockLow THEN
	MeasurementNotOk:=FALSE;
END_IF;

(*elemento nella stazione di  verifica*)
IF RotaryMakerInPositionVerification AND ElementRotaryCharged AND NOT VacuumGeneratorBlockHigh THEN	(*Quando il braccio e' in posizione di verifica, l'elemento in questa posizione diventa visibile*)
	ElementVerificationCharged:=TRUE;
	ElementVerificationRed:=ElementRotaryRed;
	ElementVerificationBlack:=ElementRotaryBlack;
	ElementVerificationSilver:=ElementRotarySilver;
	ElementVerificationOverturned:=ElementRotaryOverturned;
	ElementVerificationShort:=ElementRotaryShort;
	ElementRotaryCharged:=FALSE;
	ReadyForRotaryMaker:=FALSE;
END_IF;

(*sensore di presenza alla base dell'ascensore*)
IF NOT (ReadyLoadForVerificationBlockHigh OR ReadyLoadForVerificationBlockLow) AND NOT RotaryMakerInPositionVerificationBlockLow  THEN
	IF LiftPosition<=8 AND ElementVerificationCharged OR (RotaryMakerInPositionVerification AND ElementRotaryCharged)  THEN
			ReadyLoadForVerification:=TRUE;  (*sensore di presenza alla base dell'ascensore*)
		ELSE ReadyLoadForVerification:=FALSE;
	END_IF;
END_IF;
(*caso RotaryMakerInPositionVerificationBlockLow*)
IF NOT (ReadyLoadForVerificationBlockHigh OR ReadyLoadForVerificationBlockLow) AND RotaryMakerInPositionVerificationBlockLow THEN
	IF LiftPosition<=8 AND ElementRotaryCharged AND RotaryPosition=180 THEN
			ReadyLoadForVerification:=TRUE;
	END_IF;
	IF  LiftPosition>8 OR ToExtendCylinderOfExtractionVsGuide THEN
			ReadyLoadForVerification:=FALSE;
	END_IF;
END_IF;

(*sensore di presenza alla base dell'ascensore guasto: *)
IF ReadyLoadForVerificationBlockHigh THEN
	ReadyLoadForVerification:=TRUE;
END_IF;
IF ReadyLoadForVerificationBlockLow THEN
	ReadyLoadForVerification:=FALSE;
END_IF;

(*visualizzazione ElementoMisura*)
IF CylinderUpToMeasureLoad (*.........AND NOT ElementOneTableCharged.........*) AND ElementVerificationCharged AND NOT CylinderUpToMeasureLoadBlockHigh THEN	(*Quando l'ascensore arriva in cima viene reso visibile l'elemento in questa posizione e invisibile quello in posizione di verifica*)
	ElementMeasureCharged:=TRUE;
	ElementMeasureRed:=ElementVerificationRed;
	ElementMeasureSilver:=ElementVerificationSilver;
	ElementMeasureBlack:=ElementVerificationBlack;
	ElementMeasureOverturned:=ElementVerificationOverturned;
	ElementMeasureShort:=ElementVerificationShort;
	ElementVerificationCharged:=FALSE;
END_IF;

(*cuscinetto d'aria*)
IF NOT AirCushionBlock THEN
	IF AirCushion THEN
			 AirCushionVis:=TRUE; (*visualizzazione cuscinetto*)
		ELSE IF NOT AirCushionBlockHigh THEN
				AirCushionVis:=FALSE;
			END_IF;
	END_IF;
END_IF;

(*Elemento cuscinetto d'aria*)
IF (AirCushionVis OR AirCushionBlock) AND (CylinderOfExtractionPosition=80)  THEN	(*Quando e' attiva l'aria dello scivolo e ho spinto il pezzo, viene reso visibile l'elemento in questa posizione e invisibile quello in posizione di misura*)
	ElementAirCharged:=TRUE;
	ElementAirRed:=ElementMeasureRed;
	ElementAirSilver:=ElementMeasureSilver;
	ElementAirBlack:=ElementMeasureBlack;
	ElementAirOverturned:=ElementMeasureOverturned;
	ElementMeasureCharged:=FALSE;
END_IF;

IF (AirCushionVis AND NOT AirCushionBlock) AND (CylinderOfExtractionPosition=80)  AND MovementElementAir<=20  THEN (*Quando l'aria  dello scivolo e' attiva e il pezzo � stato spinto, il pezzo va verso la tavola rotante*)
		MovementElementAir:=MovementElementAir+3;
	ELSE MovementElementAir:=0;
END_IF;

IF (MovementElementAir>=20)
THEN	MovementElementAir:=20;
		ElementAirVsRotaryTable:=TRUE;
		AirCushion:=FALSE;
		ElementAirCharged:=FALSE;
END_IF;

IF NOT Alligneed AND NOT ElementMeasureCharged
THEN	ElementAirRed:=FALSE;
		ElementAirSilver:=FALSE;
		ElementAirBlack:=FALSE;
		ElementMeasureRed:=FALSE;
		ElementMeasureSilver:=FALSE;
		ElementMeasureBlack:=FALSE;
END_IF;

(*---------------------------------------------------------*)
IF Reset OR Remove OR FaultDetected THEN  (*rimozione di tutti i pezzi presenti sulle macchine tramite pulsante o reset del sistema*)
	ElementOneCharged:=FALSE;
	ElementOneRed:=FALSE;
	ElementOneSilver:=FALSE;
	ElementOneBlack:=FALSE;
	ElementOneOverturned:=FALSE;
	ElementOneShort:=FALSE;
	ElementTwoCharged:=FALSE;
	ElementTwoRed:=FALSE;
	ElementTwoSilver:=FALSE;
	ElementTwoBlack:=FALSE;
	ElementTwoOverturned:=FALSE;
	ElementTwoShort:=FALSE;
	ElementThreeCharged:=FALSE;
	ElementThreeRed:=FALSE;
	ElementThreeSilver:=FALSE;
	ElementThreeBlack:=FALSE;
	ElementThreeOverturned:=FALSE;
	ElementThreeShort:=FALSE;
	ElementFourCharged:=FALSE;
	ElementFourRed:=FALSE;
	ElementFourSilver:=FALSE;
	ElementFourBlack:=FALSE;
	ElementFourOverturned:=FALSE;
	ElementFourShort:=FALSE;
	ElementFiveCharged:=FALSE;
	ElementFiveRed:=FALSE;
	ElementFiveSilver:=FALSE;
	ElementFiveBlack:=FALSE;
	ElementFiveOverturned:=FALSE;
	ElementFiveShort:=FALSE;
	ElementSixCharged:=FALSE;
	ElementSixRed:=FALSE;
	ElementSixSilver:=FALSE;
	ElementSixBlack:=FALSE;
	ElementSixOverturned:=FALSE;
	ElementSixShort:=FALSE;
	ElementSevenCharged:=FALSE;
	ElementSevenRed:=FALSE;
	ElementSevenSilver:=FALSE;
	ElementSevenBlack:=FALSE;
	ElementSevenOverturned:=FALSE;
	ElementSevenShort:=FALSE;
	ElementEightCharged:=FALSE;
	ElementEightRed:=FALSE;
	ElementEightSilver:=FALSE;
	ElementEightBlack:=FALSE;
	ElementEightOverturned:=FALSE;
	ElementEightShort:=FALSE;
	ElementWaitingCharged:=FALSE;
	ElementWaitingOverturned:=FALSE;
	ElementWaitingShort:=FALSE;
	ElementRotaryCharged:=FALSE;
	ElementRotaryOverturned:=FALSE;
	ElementRotaryShort:=FALSE;
	ElementVerificationCharged:=FALSE;
	ElementVerificationOverturned:=FALSE;
	ElementVerificationShort:=FALSE;
	ElementMeasureCharged:=FALSE;
	ElementMeasureOverturned:=FALSE;
	ElementMeasureShort:=FALSE;
	ElementAirCharged:=FALSE;
	ElementAirOverturned:=FALSE;
	ElementOneTableCharged:=FALSE;
	ElementOneTableOverturned:=FALSE;

	ElementAirVsRotaryTable:=FALSE;
	ElementAirRed:=FALSE;
	ElementAirBlack:=FALSE;
	ElementAirSilver:=FALSE;
END_IF;               D   , ��� +t           Processing_PRG ��d	��d      ��������        �  PROGRAM Processing_PRG
VAR
	OperationType : INT := 666; (* To menage the init phase of Generic Devices *)
	state: Processing_States := Processing_ready_to_initialize;
	Null_Data : Data_Handler;
END_VAR


(* Handler - Between MAKING_Prg and PROCESSING_Prg *)
VAR_EXTERNAL
	Processing_Handler:	Subsystem_Handler;
	Rotary_Data:		Data_Handler;
	Inspection_Data:		Data_Handler;
	Drilling_Data:		Data_Handler;
	Expelling_Data:		Data_Handler;
END_VAR
(* Handler - Between PROCESSING_Prg and INSPECTION_UNIT_Prg *)
VAR_EXTERNAL
	Inspection_Handler:	Subsystem_Handler;
	Inspection_SubData:	Data_Handler;
END_VAR
(* Handler - Between PROCESSING_Prg and DRILLING_UNIT_Prg *)
VAR_EXTERNAL
	Drilling_Handler:	Subsystem_Handler;
END_VAR



(* GDs - Instances and  Handler request*)
VAR
   RotaryTable : 	Generic_Device;
   RotaryTable_enable_request : 	BOOL;
   RotaryTable_disable_request :	BOOL;
   RotaryTable_not_initialized : BOOL;

   ExpellingLever : 	Generic_Device;
   ExpellingLever_enable_request :	BOOL;
   ExpellingLever_disable_request :	BOOL;
   ExpellingLever_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(* RotaryTable - DANR_DF *)
     enable_RotaryTable : BOOL;
     disable_RotaryTable : BOOL;
     RotaryTable_enabled : BOOL;
     RotaryTable_EnabledSensorFault : BOOL;
     RotaryTable_disabled : BOOL;
     RotaryTable_DisabledSensorFault : BOOL;
     RotaryTable_fault :BOOL;
     RotaryTable_ActuatorFault : BOOL;

	(* ExpellingLever - SA_NF *)
     enable_ExpellingLever : BOOL;
     ExpellingLever_ActuatorFault : BOOL;

	(* Pure sensors *)
	AvailableLoadForWorkingStation_Logical : BOOL;
END_VARl  (*** FSM ***)
IF Processing_Handler.ImmediateStop THEN
	Drilling_Handler.ImmediateStop:=TRUE;
	Inspection_Handler.ImmediateStop:=TRUE;
ELSE
	Drilling_Handler.ImmediateStop:=FALSE;
	Inspection_Handler.ImmediateStop:=FALSE;

	CASE state OF
	
	Processing_ready_to_initialize:
	   IF Processing_Handler.Initialize THEN
		OperationType := INIT;
		Drilling_Handler.Initialize := TRUE;
		Inspection_Handler.Initialize := TRUE;
	
		state := Processing_initializing;
	   END_IF;
	
	Processing_initializing:
	   IF (NOT Drilling_Handler.Initialize AND NOT Inspection_Handler.Initialize AND NOT RotaryTable_not_initialized AND NOT ExpellingLever_not_initialized) THEN
		OperationType := RUN;
		Processing_Handler.Initialize := FALSE;
	
		state := Processing_ready_to_enable;
	   END_IF;
	
	Processing_ready_to_enable:
	   IF ((Processing_Handler.Enable) AND NOT(INT_TO_BOOL(Rotary_Data.ID) XOR AvailableLoadForWorkingStation_Logical)) THEN
	       RotaryTable_enable_request := TRUE;
	
	       state := Rotary_enabling;
	   END_IF;
	
	Rotary_enabling:
	   IF NOT RotaryTable_enable_request THEN
	       RotaryTable_disable_request := TRUE;
	
	       state := Rotary_disabling;
	   END_IF;
	
	Rotary_disabling:
	IF NOT RotaryTable_disable_request THEN
		Expelling_Data := 	Drilling_Data;
		Drilling_Data := 		Inspection_Data;
		Inspection_Data := 	Rotary_Data;
		Rotary_Data := 		Null_Data;
	
		IF INT_TO_BOOL(Inspection_Data.ID) THEN
			Inspection_SubData := Inspection_Data;
			Inspection_Handler.Enable := TRUE;
		END_IF;
	
		IF INT_TO_BOOL(Drilling_Data.ID) AND NOT Drilling_Data.Discard THEN
			Drilling_Handler.Enable := TRUE;
		END_IF;
		
		state := Units_enabling;
	END_IF
	
	Units_enabling:
	IF NOT(Inspection_Handler.Enable) AND NOT(Drilling_Handler.Enable) THEN
		IF INT_TO_BOOL(Inspection_Data.ID) THEN
			Inspection_Data.Orientation := Inspection_SubData.Orientation;
			Inspection_Handler.Disable := TRUE;
	   	END_IF;
	
	   	IF INT_TO_BOOL(Drilling_Data.ID) AND NOT Drilling_Data.Discard THEN
	     		Drilling_Handler.Disable := TRUE;
	
		END_IF;
	
	       state := Units_disabling;
	END_IF;
	
	Units_disabling:
	   IF NOT(Inspection_Handler.Disable) AND NOT(Drilling_Handler.Disable) THEN
	       Processing_Handler.Enable := FALSE;
	
	       state := Processing_ready_to_disable;
	   END_IF;
	
	Processing_ready_to_disable:
	   IF (Processing_Handler.Disable) AND (INT_TO_BOOL(Expelling_Data.ID)) THEN
	       ExpellingLever_enable_request := TRUE;
	
	       state := Expelling_enabling;
	   END_IF;
	
	   IF (Processing_Handler.Disable) AND NOT (INT_TO_BOOL(Expelling_Data.ID)) THEN
	       Processing_Handler.Disable := FALSE;
	
	       state := Processing_ready_to_enable;
	   END_IF;
	
	Expelling_enabling:
	   IF NOT ExpellingLever_enable_request THEN
	       ExpellingLever_disable_request := TRUE;
	
	       state := Expelling_disabling;
	   END_IF;
	
	Expelling_disabling:
	   IF NOT ExpellingLever_disable_request THEN
	       Processing_Handler.Disable := FALSE;
	
	       state := Processing_ready_to_enable;
	   END_IF;
	END_CASE;

END_IF



(*** GENERIC DEVICES ***)
RotaryTable.DeviceOperation := OperationType;
RotaryTable.DeviceClock := TRUE;
RotaryTable.DeviceDiagnosticsEnabled := TRUE;
RotaryTable.DeviceEnablePreset := FALSE;
RotaryTable.DeviceEnabledSensor := RotaryTable_enabled;
RotaryTable.DeviceDisabledSensor := RotaryTable_disabled;
RotaryTable.DeviceEnableTime := RotaryTable_EnableTime;
RotaryTable.DeviceDisableTime := RotaryTable_DisableTime;
RotaryTable.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_DA_NO_RETAIN;
RotaryTable(DeviceEnableRequest := RotaryTable_enable_request, DeviceDisableRequest := RotaryTable_disable_request );
enable_RotaryTable := RotaryTable.EnableDevice;
RotaryTable_not_initialized:=RotaryTable.DeviceNotInitialized;
disable_RotaryTable := RotaryTable.DisableDevice;
RotaryTable_ActuatorFault := RotaryTable.DeviceActuatorFault;
RotaryTable_EnabledSensorFault := RotaryTable.DeviceEnabledSensorFault;
RotaryTable_DisabledSensorFault := RotaryTable.DeviceDisabledSensorFault;
RotaryTable_fault := RotaryTable.DeviceFault;

ExpellingLever.DeviceOperation := OperationType;
ExpellingLever.DeviceClock := TRUE;
ExpellingLever.DeviceDiagnosticsEnabled := TRUE;
ExpellingLever.DeviceEnablePreset := FALSE;
ExpellingLever.DeviceEnableTime := ExpellingLever_EnableTime;
ExpellingLever.DeviceDisableTime := ExpellingLever_DisableTime;
ExpellingLever.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
ExpellingLever(DeviceEnableRequest := ExpellingLever_enable_request, DeviceDisableRequest := ExpellingLever_disable_request );
enable_ExpellingLever := ExpellingLever.EnableDevice;
ExpellingLever_not_initialized:=ExpellingLever.DeviceNotInitialized;
ExpellingLever_ActuatorFault := ExpellingLever.DeviceActuatorFault;               �  , ��� H           Pulsantiera ��d	��d      ��������        !   PROGRAM Pulsantiera
VAR
END_VARt  (* Pulsantiera di controllo *)
IF EnableVirtualBox THEN  (* se la pulsantiera virtuale � abilitata le uscite saranno quelle della pulsantiera virtuale *)
	ToWorkBlackLoad:=ToWorkBlackLoadPuls;
	ToWorkBlackOrRedLoad:=ToWorkBlackOrRedLoadPuls;
	FullWarehouse:=FullWarehousePuls;
	UpsideDownLoadRemovedInExpelling:=UpsideDownLoadRemovedInExpellingPuls;
	FullWarehouseInAssemblyStation:=FullWarehouseInAssemblyStationPuls;
ELSE (*altrimenti uso la pulsantiera fisica*)
	ToWorkBlackLoad:=ToWorkBlackLoadPin;
	ToWorkBlackOrRedLoad:=ToWorkBlackOrRedLoadPin;
	FullWarehouse:=FullWarehousePin;
	UpsideDownLoadRemovedInExpelling:=UpsideDownLoadRemovedInExpellingPin;
	FullWarehouseInAssemblyStation:=FullWarehouseInAssemblyStationPin;
END_IF;

IF FaultDetected
THEN
	(*SENSORI*)
	AlignementRotaryTableWithPositioningsBlockHigh:=FALSE;
	AlignementRotaryTableWithPositioningsBlockLow:=FALSE;

	AvailableLoadForWorkingStationBlockHigh:=FALSE;
	AvailableLoadForWorkingStationBlockLow:=FALSE;

	AvailableLoadInControlPositioningBlockHigh:=FALSE;
	AvailableLoadInControlPositioningBlockLow:=FALSE;

	AvailableLoadInDrillingPositioningBlockHigh:=FALSE;
	AvailableLoadInDrillingPositioningBlockLow:=FALSE;

	InControlLoadInWrongPositionToBeDrilledBlockHigh:=FALSE;
	InControlLoadInWrongPositionToBeDrilledBlockLow:=FALSE;

	DrillingUnitUpBlockHigh:=FALSE;
	DrillingUnitUpBlockLow:=FALSE;

	DrillingUnitDownBlockHigh:=FALSE;
	DrillingUnitDownBlockLow:=FALSE;

	AvailableLoadForRobotBlockHigh:=FALSE;
	AvailableLoadForRobotBlockLow:=FALSE;

	RobotInInitialPositionBlockHigh:=FALSE;
	RobotInInitialPositionBlockLow:=FALSE;

	RobotInAssemblyUnitBlockHigh:=FALSE;
	RobotInAssemblyUnitBlockLow:=FALSE;

	RobotInPistonWarehouseBlockHigh:=FALSE;
	RobotInPistonWarehouseBlockLow:=FALSE;

	RobotInSpringWarehouseBlockHigh:=FALSE;
	RobotInSpringWarehouseBlockLow:=FALSE;

	RobotInCoverWarehouseBlockHigh:=FALSE;
	RobotInCoverWarehouseBlockLow:=FALSE;

	EmptyCoverHouseInAssemblyStationBlockHigh:=FALSE;
	EmptyCoverHouseInAssemblyStationBlockLow:=FALSE;

	ToExtractCoverInAssemblyStationInRetroactivePositionBlockHigh:=FALSE;
	ToExtractCoverInAssemblyStationInRetroactivePositionBlockLow:=FALSE;

	ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh:=FALSE;
	ToExtractCoverInAssemblyStationInExtensivePositionBlockLow:=FALSE;

	PistonSelectorIsOnTheRightBlockHigh:=FALSE;
	PistonSelectorIsOnTheRightBlockLow:=FALSE;

	PistonSelectorIsOnTheLeftBlockHigh:=FALSE;
	PistonSelectorIsOnTheLeftBlockLow:=FALSE;

	ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh:=FALSE;
	ToExtractSpringInAssemblyStationInExtensivePositionBlockLow:=FALSE;

	ToExtractSpringInAssemblyStationInRetroactivePositionBlockHigh:=FALSE;
	ToExtractSpringInAssemblyStationInRetroactivePositionBlockLow:=FALSE;

	(*  ATTUATORI  *)

	RotaryTableMotorBlockHigh:=FALSE;
	RotaryTableMotorBlockLow:=FALSE;

	ToLowerCylinderToInspectLoadBlockHigh:=FALSE;
	ToLowerCylinderToInspectLoadBlockLow:=FALSE;

	DrillingUnitActiveBlockHigh:=FALSE;
	DrillingUnitActiveBlockLow:=FALSE;

	ToLowerDrillingUnitBlockHigh:=FALSE;
	ToLowerDrillingUnitBlockLow:=FALSE;

	ToLiftDrillingUnitBlockHigh:=FALSE;
	ToLiftDrillingUnitBlockLow:=FALSE;

	BlockingCylinderForwardInDrillingPositioningBlockHigh:=FALSE;
	BlockingCylinderForwardInDrillingPositioningBlockLow:=FALSE;

	ExpellingLeverActiveBlockHigh:=FALSE;
	ExpellingLeverActiveBlockLow:=FALSE;

	ToExtractSpringInAssemblyStationBlockHigh:=FALSE;
	ToExtractSpringInAssemblyStationBlockLow:=FALSE;

	PistonSelectorGoOnTheRightBlockHigh:=FALSE;
	PistonSelectorGoOnTheRightBlockLow:=FALSE;

	PistonSelectorGoOnTheLeftBlockHigh:=FALSE;
	PistonSelectorGoOnTheLeftBlockLow:=FALSE;

	ToExtractCoverInAssemblyStationForwardBlockHigh:=FALSE;
	ToExtractCoverInAssemblyStationForwardBlockLow:=FALSE;

	BlockingCylinderForwardInAssemblyStationBlockHigh:=FALSE;
	BlockingCylinderForwardInAssemblyStationBlockLow:=FALSE;

	RobotTakeBlackLoadBlockHigh:=FALSE;
	RobotTakeBlackLoadBlockLow:=FALSE;

	RobotTakeRedSilverLoadBlockHigh:=FALSE;
	RobotTakeRedSilverLoadBlockLow:=FALSE;

	RobotTakeLoadToDiascardBlockHigh:=FALSE;
	RobotTakeLoadToDiascardBlockLow:=FALSE;

	RobotGoToInitialPositionBlockHigh:=FALSE;
	RobotGoToInitialPositionBlockLow:=FALSE;

	RobotGoToSpringHouseBlockHigh:=FALSE;
	RobotGoToSpringHouseBlockLow:=FALSE;

	RobotGoToPistonHouseBlockHigh:=FALSE;
	RobotGoToPistonHouseBlockLow:=FALSE;

	RobotGoToCoverHouseBlockHigh:=FALSE;
	RobotGoToCoverHouseBlockLow:=FALSE;

	RobotTakeCurrentLoadToAssemblyBlockHigh:=FALSE;
	RobotTakeCurrentLoadToAssemblyBlockLow:=FALSE;

	RobotEngineBlockLow:=FALSE;
END_IF;               +   ,  �  |        	   Robot_PRG ��d	��d      ��������        #  PROGRAM Robot_PRG
(* Memory array to manage data memory and its operation *)
VAR_EXTERNAL
    	Memory_Data: ARRAY [1..8] OF Data_Handler;
END_VAR

(* Arrey's index *)
VAR_EXTERNAL CONSTANT
	Distribution_index:		UINT:=1;
	Testing_index:			UINT:=2;
	Rotary_index:			UINT:=3;
	Inspection_index:			UINT:=4;
	Drilling_index:			UINT:=5;
	Expelling_index:			UINT:=6;
	PickandPlace_index:		UINT:=7;
	Supply_index:			UINT:=8;
END_VAR




(* Between MAIN_Prg and ROBOT_Prg *)
VAR_EXTERNAL
	Robot_Handler : 	System_Handler;
END_VAR

(* Between ROBOT_Prg and PICKANDPLACE_Prg*)
VAR_EXTERNAL
	PickandPlace_Handler : 	Subsystem_Handler;
	PickandPlace_Data :		Data_Handler;
END_VAR

(* Between ROBOT_Prg and ASSEMBLY_PRG *)
VAR_EXTERNAL
	Assembly_Handler:		Subsystem_Handler;
	Assembly_Data:		Data_Handler;
END_VAR

(* Between ASSEMBLY_Prg and MAKING_Prg *)
VAR_EXTERNAL
	Robot_ready_to_receive:BOOL:= TRUE;
END_VAR


VAR
	state_Robot: Robot_States := Robot_ready_to_initialize;
END_VAR
VAR
	Null_Data : Data_Handler;
END_VAR�  (***FSM - ROBOT***)
IF Robot_Handler.ImmediateStop THEN
	PickandPlace_Handler.ImmediateStop:=TRUE;
	Assembly_Handler.ImmediateStop:=TRUE;
ELSE
	PickandPlace_Handler.ImmediateStop:=FALSE;
	Assembly_Handler.ImmediateStop:=FALSE;

	CASE state_Robot OF

	Robot_ready_to_initialize:
	   IF Robot_Handler.Initialize THEN
	       PickandPlace_Handler.Initialize := TRUE;
		Assembly_Handler.Initialize := TRUE;
	
	       state_Robot := Robot_Initializing;
	   END_IF;
	
	Robot_Initializing:
	   IF NOT PickandPlace_Handler.Initialize AND NOT Assembly_Handler.Initialize THEN
	       Robot_Handler.Initialize := FALSE;
	
	       state_Robot := Robot_ready_to_run;
	   END_IF;
	
	Robot_ready_to_run:
	   IF Robot_Handler.Run AND NOT Robot_ready_to_receive AND INT_TO_BOOL (Memory_Data[PickandPlace_index].ID) THEN
		PickandPlace_Data := 	Memory_Data[PickandPlace_index];
	       PickandPlace_Handler.Enable := TRUE;
	
	       state_Robot := PickandPlace_enabling;
	   END_IF;
	
	PickandPlace_enabling:
	   IF NOT PickandPlace_Handler.Enable THEN
	       PickandPlace_Handler.Disable := TRUE;
	
	       state_Robot := PickandPlace_disabling;
	   END_IF;
	
	PickandPlace_disabling: (*SE SCARTATO DEVI SALTARE IL ASSEMBLY, VEDI CON LOGICA DEL TEST PER CAPIRE CHI PULISCE IL DATO *)
	   IF NOT PickandPlace_Handler.Disable THEN
		IF NOT Memory_Data[PickandPlace_index].discard THEN
			Robot_ready_to_receive := TRUE;
			Memory_Data := 	Shift_data(PickandPlace_index, Memory_Data);
			Assembly_Data := 	Memory_Data[Supply_index];
			Assembly_Handler.Enable:= TRUE;
		
		       state_Robot := Assembly_enabling;
		ELSE
			Robot_ready_to_receive := TRUE;
			Memory_Data[PickandPlace_index].ID:=0;
			Memory_Data := 	Shift_data(PickandPlace_index, Memory_Data);
			Assembly_Data := 	Memory_Data[Supply_index];
	
			state_Robot := Robot_ready_to_run;
		END_IF;
	   END_IF;
	
	Assembly_enabling:
	   IF NOT Assembly_Handler.Enable THEN
	       Assembly_Handler.Disable := TRUE;
	
	       state_Robot := Assembly_disabling;
	   END_IF;
	
	Assembly_disabling:
	   IF NOT Assembly_Handler.Disable THEN
		Memory_Data := Save_data(Supply_index, Memory_Data, Null_Data);
	
	       state_Robot := Robot_ready_to_run;
	   END_IF;
	END_CASE;
END_IF
               e   , U� �        	   Save_data ��d	��d      ��������        �   FUNCTION Save_data : ARRAY [1..8] OF Data_Handler

VAR_INPUT
	Memory_Index : UINT;
	Memory_Data: ARRAY [1..8] OF Data_Handler;
	To_Save_Data: Data_Handler;
END_VARC   Save_data := Memory_Data;
Save_data[Memory_Index] := To_Save_Data;               d   , ) �        
   Shift_data ��d	��d      ��������        �   FUNCTION Shift_data : ARRAY [1..8] OF Data_Handler

VAR_INPUT
	Memory_Index : UINT;
	Memory_Data: ARRAY [1..8] OF Data_Handler;
END_VAR

VAR
	Empty_Data:Data_Handler;
END_VAR~   Shift_data :=Memory_Data;
Shift_data[Memory_Index+1] := Memory_Data[Memory_Index];
Shift_data[Memory_Index] := Empty_Data;
               ^   , ��U �s           Signal_Filter ��d	��d      ��������        �   FUNCTIONBLOCK Signal_Filter
VAR_INPUT
	Signal : BOOL;
	ActivationDelay : UDINT := 3;
	DeactivationDelay : UDINT := 3;
END_VAR

VAR_OUTPUT
	DelayedSignal : BOOL;
END_VAR

VAR
	Delay : UDINT := 0;
END_VAR�   IF (Signal = DelayedSignal) THEN
	IF (Signal) THEN
		Delay := DeactivationDelay;
	ELSE
		Delay := ActivationDelay;
	END_IF
END_IF

IF Delay > 0 THEN
	Delay := Delay - 1;
END_IF

IF (Delay = 0) THEN
	DelayedSignal := Signal;
END_IF               Y   , Z� L�           SignalControl_PRG ��d	��d      ��������        �  PROGRAM SignalControl_PRG
(**** LOCAL****)
VAR
(* SignalManagement_FB *)
	SignalManagement : SignalManagement;
	OutputSignals : DWORD;
	ResetEnable : BOOL;
END_VAR

 (* MESSAGGE IDs *)
VAR CONSTANT
	mCylinder_EnabledSensorFault: WORD :=1;
	mCylinder_DisabledSensorFault: WORD :=2;
	mCylinder_ActuatorFault: WORD :=3;
	mRotaryMaker_EnabledSensorFault: WORD :=4;
	mRotaryMaker_DisabledSensorFault: WORD :=5;
	mRotaryMaker_ActuatorFault: WORD :=6;
	mVacuumGenerator_EnabledSensorFault: WORD :=7;
	mVacuumGenerator_ActuatorFault: WORD :=8;
	mElevator_EnabledSensorFault: WORD :=9;
	mElevator_DisabledSensorFault: WORD :=10;
	mElevator_ActuatorFault: WORD :=11;
	mExtractionCylinder_DisabledSensorFault: WORD :=12;
	mExtractionCylinder_ActuatorFault: WORD :=13;
	mRotaryTable_EnabledSensorFault: WORD :=14;
	mRotaryTable_DisabledSensorFault: WORD :=15;
	mRotaryTable_ActuatorFault: WORD :=16;
	mDrill_Machine_EnabledSensorFault: WORD :=17;
	mDrill_Machine_DisabledSensorFault: WORD :=18;
	mDrill_Machine_ActuatorFault: WORD :=19;
	mPistonSelector_EnabledSensorFault: WORD :=20;
	mPistonSelector_DisabledSensorFault: WORD :=21;
	mPistonSelector_ActuatorFault: WORD :=22;
	mExtractCover_EnabledSensorFault: WORD :=23;
	mExtractCover_DisabledSensorFault: WORD :=24;
	mExtractCover_ActuatorFault: WORD :=25;
	mExtractSpring_EnabledSensorFault: WORD :=26;
	mExtractSpring_DisabledSensorFault: WORD :=27;
	mExtractSpring_ActuatorFault: WORD :=28;
	mEmptyWarehouse_Logical: WORD :=29;
	mEmptyCoverHouseInAssemblyStation_Logical: WORD :=30;
END_VAR

VAR CONSTANT
(* OUTPUT Ids *)
	EMERGENCY_STOP 	: DWORD := 16#0001;
	IMMEDIATE_STOP 	: DWORD := 16#0002;
	ON_PHASE_STOP 	: DWORD := 16#0004;
END_VAR


(*** INPUT ***)
(* Between MAIN_Prg and SIGNALMANAGEMENT_Prg *)
VAR_EXTERNAL
	SignalManagement_Handler : 	System_Handler;
END_VAR

(* From InputBridge *)
VAR_EXTERNAL
	Reset_Logical : BOOL
	(* Possible MachineNotRunning *);
END_VAR

(*Fault from GD*)
VAR_EXTERNAL
	(* Cylinder - SA_DF *)
	Cylinder_EnabledSensorFault : BOOL;
	Cylinder_DisabledSensorFault : BOOL;
	Cylinder_fault :BOOL;
	Cylinder_ActuatorFault : BOOL;

	(* RotaryMaker - DA_DF *)
	RotaryMaker_EnabledSensorFault : BOOL;
	RotaryMaker_DisabledSensorFault : BOOL;
	RotaryMaker_fault :BOOL;
	RotaryMaker_ActuatorFault : BOOL;

	(* VacuumGenerator - SA_SAF *)
	VacuumGenerator_EnabledSensorFault : BOOL;
	VacuumGenerator_fault :BOOL;
	VacuumGenerator_ActuatorFault : BOOL;

	(* GDs - Sensors - Logical variables *)
	ExpulsionAirVacuum_ActuatorFault : BOOL;
END_VAR


(*** OUTPUT ***)
VAR_EXTERNAL
	ResetSignalsEnable : BOOL;

	ImmediateStop_Signal : BOOL;
	OnPhaseStop_Signal : BOOL;
END_VAR



(*** From library FB ***)
VAR_EXTERNAL CONSTANT
(* Signal type e relativa sua maschera *)
	SIGNAL_TYPE_MASK : WORD := 00007; (*ultimi 3 bit*)
	ALARM : WORD := 00001;
	ANOMALY : WORD := 00002;
	WARNING : WORD := 00003;
	INFORMATION : WORD := 00004;

(* To clean the variables *)
	NONE : DWORD := 16#0000;

 (*Dimensione massima dell'arrey contenente i segnali.
    Ovvero il numero massimo del "SignalCode" da associare ai segnali, generati dal file Python*)
	N :INT:=32;
END_VAR

(*Light signals*)
VAR CONSTANT
	LIGHT_EMPTYWAREHOUSE : DWORD := 16#0008;
	LIGHT_EMPTYCOVERHOUSE : DWORD := 16;
END_VAR

VAR_EXTERNAL
	LightEmptyWarehouseLogical: BOOL;
	LightEmptyCoverhouseLogical: BOOL; (*in assembly station*)
END_VAR�   ResetEnable := SignalManagement.ResetEnable;
ResetSignalsEnable :=ResetEnable; (* AND MachineNotRunning*)

IF (SignalManagement_Handler.Initialize) THEN
	SignalManagement (OperationType_States := START_CONFIGURATION); (* autonumeration selection *)
	SignalManagement.OperationType_States := CONFIGURATION;
ELSIF ( SignalManagement_Handler.Run ) THEN
	SignalManagement (OperationType_States := START_GENERATION, Reset := Reset_Logical); (* autonumeration selection *)
	SignalManagement.OperationType_States := GENERATION;
END_IF;

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mCylinder_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Cylinder_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mCylinder_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Cylinder_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mCylinder_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Cylinder_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryMaker_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryMaker_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryMaker_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryMaker_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryMaker_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryMaker_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mVacuumGenerator_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := VacuumGenerator_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mVacuumGenerator_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := VacuumGenerator_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mElevator_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Elevator_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mElevator_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Elevator_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mElevator_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Elevator_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractionCylinder_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractionCylinder_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractionCylinder_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractionCylinder_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryTable_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryTable_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryTable_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryTable_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mRotaryTable_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := RotaryTable_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mDrill_Machine_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Drill_Machine_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mDrill_Machine_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Drill_Machine_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mDrill_Machine_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := Drill_Machine_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mPistonSelector_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := PistonSelector_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mPistonSelector_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := PistonSelector_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mPistonSelector_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := PistonSelector_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractCover_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractCover_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractCover_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractCover_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractCover_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractCover_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractSpring_EnabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractSpring_EnabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractSpring_DisabledSensorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractSpring_DisabledSensorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := ALARM OR AUTO_CONDITIONED_RESET,
SignalCode := mExtractSpring_ActuatorFault,
SignalOutput := IMMEDIATE_STOP,
ActivationSignal := ExtractSpring_ActuatorFault,
AutoResetSignal := ImmediateStop_Logical);

SignalManagement
(SignalType := WARNING OR AUTO_RESET,
SignalCode := mEmptyWarehouse_Logical,
SignalOutput := LIGHT_EMPTYWAREHOUSE,
ActivationSignal := EmptyWarehouse_Logical);

SignalManagement
(SignalType := WARNING OR AUTO_RESET,
SignalCode := mEmptyCoverHouseInAssemblyStation_Logical,
SignalOutput := LIGHT_EMPTYCOVERHOUSE,
ActivationSignal := EmptyCoverHouseInAssemblyStation_Logical);


IF (SignalManagement_Handler.Run) THEN
	SignalManagement (OperationType_States := RUN_SIGNALMANAGEMENT); (* outputs selection *)
END_IF;

OutputSignals := SignalManagement.SignalOutputs;
ImmediateStop_Signal := ((OutputSignals AND IMMEDIATE_STOP) = IMMEDIATE_STOP);
OnPhaseStop_Signal := ((OutputSignals AND ON_PHASE_STOP) = ON_PHASE_STOP);

(*Light signals*)
LightEmptyWarehouseLogical := ((OutputSignals AND LIGHT_EMPTYWAREHOUSE) = LIGHT_EMPTYWAREHOUSE);
LightEmptyCoverhouseLogical := ((OutputSignals AND LIGHT_EMPTYCOVERHOUSE) = LIGHT_EMPTYCOVERHOUSE);

               o   ,     �           SignalManagement ��d	��d      ��������        r  FUNCTION_BLOCK SignalManagement
VAR_INPUT
	OperationType_States : SignalManagement_States := INIT_SIGNALMANAGEMENT; (*States*)
	SignalType : WORD; (* Usato come Array di dimensione 16, i cui primi 3 bit rappresentano i possibili tipi di segnale *)
	SignalCode : WORD; (* Codice del segnale di ingresso viene usato come indice nell'Array 'Signals' *)
	SignalOutput : DWORD; (* Array di bit il cui ogni bit identifica un reazione al dato segnale. Verr� usato per settare il relativo bit nella variabile di 'SignalOutputs' *)
	ActivationSignal : BOOL; (* Segnale di ingresso  *)
	AutoResetSignal : BOOL; (* ??? *)

	(* START_GENERATION *)
	Reset:BOOL; (* Possibile segnale di input dal bridge fisico*)
	KeyReset: BOOL;  (* Possibile segnale di input dal bridge fisico*)

END_VAR

VAR_OUTPUT
	(* START_GENERATION *)
	ResetEnable : BOOL; (*Richiesta di reset. Accensione del led di reset, � possibile effetuare il reset *)
	SignalOutputs : DWORD; (* Un Array di dimensione 32, i cui primi 3 bit rappresentano i possibili STOP *)
END_VAR

VAR (* inputs/outputs *)
	(* Number of signal initialized in "START_CONFIGURATION states" *)
	NumberOfAlarms : INT;
	NumberOfAnomalies : INT;
	NumberOfWarnings : INT;
	NumberOfInformation : INT;

	(* START_GENERATION *)
	ResetOld:BOOL; (*Per mantenere la memoria del segnale e effetuare il risign edge*)
	ResetActivation : BOOL; (* Rising edge del segnale di reset *)

	KeyResetOld: BOOL; (*Per mantenere la memoria del segnale e effetuare il risign edge*)
	AuxResetActivation : BOOL;  (* Rising edge del segnale di keyreset *)

	(* Usato in GENERATION per definire l'idice del segnale *)
	Index: INT;
	Current_index: INT; (*Auxiliary index for auto-conditioned reset*)
	BaseIndex: INT;
	NumberOfActiveAlarms : INT;
	NumberOfActiveAnomalies : INT;
	NumberOfActiveWarning : INT;
	NumberOfActiveInformation : INT;
	Condition: BOOL; (* Serve per differenziare fra Allarms o Information, i primi vanno resettati *)

	(* ARRAYS *)
	Signals : ARRAY[1..N] OF BOOL; (* Insieme di tutti i segnali, ove il suo indice � il SignalCode dato in input *)
	ActiveSignalCodes : ARRAY[1..N] OF WORD; (* Salviamo in ordine di attivazione i codici dei segnali attivati *)
	i: INT := 2; (* Usato come puntatore in un ciclo FOR nel codice *)

END_VAR


VAR_EXTERNAL CONSTANT
(* Signal type e relativa sua maschera *)
	SIGNAL_TYPE_MASK : WORD := 00007; (*ultimi 3 bit*)
	ALARM : WORD := 00001;
	ANOMALY : WORD := 00002;
	WARNING : WORD := 00003;
	INFORMATION : WORD := 00004;

	(* Signal Reset Definitions *)
	SIGNAL_RESET_MASK : WORD := 15360; (*10 A 13 bit*)
	AUTO_RESET : WORD := 1024;
	AUTO_CONDITIONED_RESET : WORD := 08192;

(* To clean the variables *)
	NONE : DWORD := 16#0000;

 (*Dimensione massima dell'arrey contenente i segnali.
    Ovvero il numero massimo del "SignalCode" da associare ai segnali, generati dal file Python*)
	N :INT:=32;
END_VAR








(**** USEFULL TIPS FROM PALLI TO EXPANDING THE LYBRARY ***)
(*
VAR
	ResetSignalsEnable : BOOL;
	AuxResetEnable : BOOL;
	SignalPriorityCode : WORD;

	SignalRequest : BOOL;
	SignalTypeAlarm : BOOL;
	SignalTypeAnomaly : BOOL;
	SignalTypeWarning : BOOL;
	SignalTypeInformation : BOOL;
	SignalAckRequest : BOOL;
	FirstSignalIndex : INT;
	LastSignalIndex : INT;

	ActiveSignalRequest : BOOL;
	ActiveSignalRequestAck : BOOL;
	FirstActiveSignalIndex : INT;
	LastActiveSignalIndex : INT;

	SignalManagementError : BOOL;
END_VAR

VAR_EXTERNAL (* CONFIGURATION *)
	ConditionedResetSignals : BOOL;
	TimePrioritySignals : BOOL;
	SignalCodeDefault : WORD;
	SignalCodeImpossible : WORD;
	SignalCodeError : WORD;
END_VAR
*)�  CASE OperationType_States OF

	START_CONFIGURATION: (* Reset signals counts *)
		NumberOfAlarms := 0;
		NumberOfAnomalies := 0;
		NumberOfWarnings := 0;
		NumberOfInformation := 0;

	CONFIGURATION: (* counting signals *)
		CASE (SignalType AND SIGNAL_TYPE_MASK) OF
			ALARM:
				NumberOfAlarms := NumberOfAlarms + 1;
			ANOMALY:
				NumberOfAnomalies := NumberOfAnomalies + 1;
			WARNING:
				NumberOfWarnings := NumberOfWarnings + 1;
			INFORMATION:
				NumberOfInformation := NumberOfInformation + 1;
		END_CASE;

	START_GENERATION: (* Prepare for generation, eseguita solo una volta *)
		ResetEnable := FALSE;
		SignalOutputs := NONE;
		(* Input dal SignalControl_Prg, elaborazione dei segnali per essere impulsivi *)
		ResetActivation := (Reset AND NOT ResetOld); (* Risign edge del segnale di Reset. (impulso) *)
		ResetOld := Reset;
		AuxResetActivation := (KeyReset AND NOT KeyResetOld);
		KeyResetOld:= KeyReset;

	GENERATION: (* Eseguito per ogni segnale, stiamo seguendo la slide 15 " Dynamics Signal e selective reset " *)
		CASE (SignalType AND SIGNAL_TYPE_MASK) OF
			ALARM:
				Index := NumberOfActiveAlarms; (* Numeri di allarmi attivi fino ad ora *)
				BaseIndex := 0;
				Condition := TRUE;
			ANOMALY:
				Index := NumberOfActiveAnomalies;
				BaseIndex := NumberOfAlarms;
				Condition := TRUE;
			WARNING:
				Index := NumberOfActiveWarning;
				BaseIndex := NumberOfAlarms + NumberOfAnomalies;
				Condition := FALSE;
			INFORMATION:
				Index := NumberOfActiveInformation;
				BaseIndex := NumberOfAlarms + NumberOfAnomalies + NumberOfWarnings;
				Condition := FALSE;
			END_CASE;

		(* Se il segnale � attivo ma non era ancora stato salvato in memoria,
			lo aggiorni nella memoria 'Signals' per segnare che � gi� stato valutato,
			lo insrisci nella memoria 'ActiveSignalCodes' *)
		IF ActivationSignal AND NOT Signals[SignalCode] THEN
			Signals[SignalCode] := TRUE;
			Index := Index + 1;
			ActiveSignalCodes[BaseIndex + Index] := SignalCode;
		END_IF;

		(* Se
			- Il segnale � un Allarme o un Anomalia
			- Il segnale � stato attivo e quindi salvato nel vettoro 'Signals'
			- Il segnale non � pi� attivo (ActivationSignal input variable )
			- Il segnale � il primo nella lista dei segnali attivi della sua categoria

		Allora:
			- Avverto che � possibile resettarlo

		Nel caso abbia condizioni di auto reset oppure auto_provisional_rest  lo devo fare qui*)
		IF Condition AND Signals[SignalCode] AND NOT ActivationSignal AND (ActiveSignalCodes[BaseIndex + 1] = SignalCode) THEN
			ResetEnable := TRUE;
		END_IF;

		(* Se
			- La condizione dell'autoreset del segnale � verificata 'AutoResetSignal'
			- Il segnale � stato attivo e quindi salvato nel vettoro 'Signals'
			- Il segnale non � pi� attivo (ActivationSignal input variable )
			- Il segnale NON � il primo nella lista dei segnali attivi della sua categoria

		Allora:
			- Resetto il segnale in automatico e scalo tutti gli altri nella lista*)
		CASE (SignalType AND SIGNAL_RESET_MASK) OF

			AUTO_RESET:
				IF (Signals[SignalCode] AND NOT ActivationSignal) THEN
					Signals[SignalCode] := FALSE;
					FOR i := 2 TO (Index+1) DO
						IF ActiveSignalCodes[BaseIndex + i-1] = SignalCode THEN
							Current_index := i;
							EXIT;
						END_IF
					END_FOR;
					FOR i := Current_index  TO Index DO
						ActiveSignalCodes[BaseIndex + i-1] := ActiveSignalCodes[BaseIndex + i];
					END_FOR;
					ActiveSignalCodes[BaseIndex + Index] := 0;
					Index := Index - 1;
				END_IF

			AUTO_CONDITIONED_RESET:
				(* Non voglio autoresettare il primo ma tutti gli altri dopo di lui se vanno down *)
				(* IF AutoResetSignal AND Signals[SignalCode] AND NOT ActivationSignal AND NOT(ActiveSignalCodes[BaseIndex + 1] = SignalCode) THEN *)

				(* voglio autoresettare tutti eccetto l'ultimo attivo *)
				(* IF (AutoResetSignal AND Signals[SignalCode] AND NOT ActivationSignal AND NOT(ActiveSignalCodes[BaseIndex + 2] = 0)) THEN *)

				IF (AutoResetSignal AND Signals[SignalCode] AND NOT ActivationSignal AND NOT ((Index) = 1)) THEN
					Signals[SignalCode] := FALSE;
					FOR i := 2 TO (Index+1) DO
						IF ActiveSignalCodes[BaseIndex + i-1] = SignalCode THEN
							Current_index := i;
							EXIT;
					END_IF
					END_FOR;
					FOR i := Current_index  TO Index DO
					ActiveSignalCodes[BaseIndex + i-1] := ActiveSignalCodes[BaseIndex + i];
					END_FOR;
					ActiveSignalCodes[BaseIndex + Index] := 0;
					Index := Index - 1;
				END_IF
		END_CASE
		(* Se:
			- � stato premuto il tasto reset (rising edge del segnale) OPPURE Il segnale � un warning o in'informazione
			- Il segnale � stato attivo e quindi salvato nel vettoro 'Signals'
			- Il segnale non � pi� attivo (ActivationSignal input variable )
			- Il segnale � il primo nella lista dei segnali attivi della sua categoria

		Allora:
			- Disattivo il segnale dalla memoria dei 'Signals'
			- Shifto la memoria di 'ActiveSignalCodes' e cancello l'ultimo segnale della sua categoria
			- Aggiorno l'idice, sar� diminuito di uno
			- Ora possiamo scegliere se mantenere attivo il rising adge in modo da fare un comulative reset oppure no

		Se volessi inserire il KeyReset lo faccio qui*)
		IF (ResetActivation OR NOT Condition) AND Signals[SignalCode] AND NOT ActivationSignal AND (ActiveSignalCodes[BaseIndex + 1] = SignalCode) THEN
			Signals[SignalCode] := FALSE;
			FOR i := 2 TO Index DO
				ActiveSignalCodes[BaseIndex + i-1] := ActiveSignalCodes[BaseIndex + i];
			END_FOR;
			ActiveSignalCodes[BaseIndex + Index] := 0;
			Index := Index - 1;
			ResetActivation := FALSE; (* COMMENT THIS LINE TO GO BACK TO CUMULATIVE RESET *)
		END_IF;

		(* Aggiorno il numero di segnali attivi *)
		CASE (SignalType AND SIGNAL_TYPE_MASK) OF
			ALARM:
				NumberOfActiveAlarms := Index;
			ANOMALY:
				NumberOfActiveAnomalies := Index;
			WARNING:
				NumberOfActiveWarning := Index;
			INFORMATION:
				NumberOfActiveInformation := Index;
		END_CASE;

		(* Se il segnale � attivo, cio� non � stato disattivato, setto a TRUE il suo relativo bit di output *)
		IF Signals[SignalCode] THEN
			SignalOutputs := SignalOutputs OR SignalOutput;
		END_IF;

(*
RUN_SIGNALMANAGEMENT: Mostro il primo segnale attivo ovvero quello che verr� resettato, inrefaccia huomo macchina
*)

END_CASE;               i   , n ~           Testing_colour ��d	��d      ��������        O   FUNCTION Testing_colour : BOOL
VAR_INPUT
	Color:BOOL;
	Height:BOOL;
END_VAR>  (*sensore di misuarazione altezza. Uscita del misuratore: 1 pezzo alto, 0 pezzo basso*)
(*Colore: true:grigio/rosso - false: nero*)
IF NOT Height AND Color THEN
	Testing_colour:=TRUE; (*da scartare: grigio + pezzo basso*)
ELSE
	Testing_colour:=FALSE; (*workpiece ok: rosso/grigio + alto OR nero + basso*)
END_IF;               :   , n � �           Testing_orientation ��d	��d      ��������        K   FUNCTION Testing_orientation : BOOL
VAR_INPUT
	Orientation:BOOL;
END_VAR�   (*SENSORE per il rilevamento del corretto orientamento della base*)
(* True se orientamento corretto*)
Testing_orientation := NOT Orientation;               X   , np           Testing_PRG ��d	��d      ��������        g  PROGRAM Testing_PRG
(* Da ELIMINARE*)
VAR
	OperationType : INT := 0;
END_VAR

VAR
	state: Testing_States := Testing_ready_to_initialize;
END_VAR

(* Between MACHINE_Prg and TESTING_Prg *)
VAR_EXTERNAL
	Testing_Handler:Subsystem_Handler;
	Testing_Data:Data_Handler;
END_VAR

(* GDs - Instances and  Handler request*)
VAR
	Elevator : Generic_Device;
	Elevator_enable_request : BOOL;
	Elevator_disable_request : BOOL;
	Elevator_not_initialized : BOOL;

	ExtractionCylinder : Generic_Device;
	ExtractionCylinder_enable_request : BOOL;
	ExtractionCylinder_disable_request : BOOL;
	ExtractionCylinder_not_initialized : BOOL;

	AirCushion : Generic_Device;
	AirCushion_enable_request : BOOL;
	AirCushion_disable_request : BOOL;
	AirCushion_not_initialized : BOOL;
END_VAR

(* GDs - Actuators and Sensors*)
VAR_EXTERNAL
	(*Elevator - DA_DF *)
     enable_Elevator : BOOL;
     disable_Elevator : BOOL;
     Elevator_enabled : BOOL;
     Elevator_disabled : BOOL;
     Elevator_EnabledSensorFault : BOOL;
     Elevator_DisabledSensorFault : BOOL;
     Elevator_fault :BOOL;
     Elevator_ActuatorFault : BOOL;

	(*ExtractionCylinder - SA_SDF *)
     enable_ExtractionCylinder : BOOL;
     ExtractionCylinder_disabled : BOOL;
     ExtractionCylinder_DisabledSensorFault : BOOL;
     ExtractionCylinder_fault :BOOL;
     ExtractionCylinder_ActuatorFault : BOOL;

	(*AirCushion - SA_NF*)
     enable_AirCushion : BOOL;
     AirCushion_ActuatorFault : BOOL;

	(* Pure sensors *)
	ReadyLoadForVerificationLogical:BOOL;
	MeasurementNotOkLogical:BOOL;
	ColourMeasurementLogical:BOOL;
END_VAR
U  (*FSM*)
IF NOT Testing_Handler.ImmediateStop THEN
	CASE state OF
	
	Testing_ready_to_initialize:
	   IF Testing_Handler.Initialize THEN
	       OperationType := INIT;
	
	       state := Testing_initializing;
	   END_IF;
	
	Testing_initializing:
	   IF (NOT Elevator_not_initialized AND NOT ExtractionCylinder_not_initialized AND NOT AirCushion_not_initialized) THEN
	       OperationType := RUN;
		Testing_Handler.Initialize := FALSE;
	
	       state := Testing_ready_to_enable;
	   END_IF;
	
	Testing_ready_to_enable:
	   IF (Testing_Handler.Enable AND ReadyLoadForVerificationLogical) THEN
		Testing_Data.Colour:= ColourMeasurementLogical; (*sensore di rilevazione colore: 0 nero, 1 rosso/metallico *)
	       Elevator_enable_request := TRUE;
	
	       state := Elevator_enabling;
	   END_IF;
	
	Elevator_enabling:
	   IF NOT Elevator_enable_request THEN
		Testing_Data.Height:=MeasurementNotOkLogical; (*sensore di misuarazione altezza. Uscita del misuratore: 1 pezzo alto, 0 pezzo basso*)
		Testing_Handler.Enable:=FALSE;
	
		state:=Testing_waiting_to_disable;
		END_IF;
	
	Testing_waiting_to_disable:
		IF Testing_Data.Discard AND Testing_Handler.Disable THEN (*da scartare*)
			Elevator_disable_request:=TRUE;
	
			state := Elevator_disabling_NOT_OK;
	
		ELSIF NOT Testing_Data.Discard AND Testing_Handler.Disable THEN (*pezzo OK*)
			AirCushion_enable_request := TRUE;
	
			state := AirCushion_enabling_OK;
		END_IF;

(*workpiece da scartare*)
Elevator_disabling_NOT_OK:
   IF NOT Elevator_disable_request THEN
       ExtractionCylinder_enable_request := TRUE;

       state := ExtractionCylinder_enabling_NOT_OK;
   END_IF;

(*scarto il pezzo*)
ExtractionCylinder_enabling_NOT_OK:
   IF NOT ExtractionCylinder_enable_request THEN
       ExtractionCylinder_disable_request := TRUE;

       state := ExtractionCylinder_disabling_NOT_OK;
   END_IF;

ExtractionCylinder_disabling_NOT_OK:
   IF NOT ExtractionCylinder_disable_request THEN
	 Testing_handler.Disable:=FALSE;

       state := Testing_ready_to_enable;
   END_IF;

(*workpiece OK*)
AirCushion_enabling_OK:
   IF NOT AirCushion_enable_request THEN
       ExtractionCylinder_enable_request := TRUE;

       state := ExtractionCylinder_enabling_OK;
   END_IF;

ExtractionCylinder_enabling_OK:
   IF NOT ExtractionCylinder_enable_request THEN
       ExtractionCylinder_disable_request := TRUE;

       state := ExtractionCylinder_disabling_OK;
   END_IF;

ExtractionCylinder_disabling_OK:
   IF NOT ExtractionCylinder_disable_request THEN
       AirCushion_disable_request := TRUE;

       state := AirCushion_disabling_OK;
   END_IF;

AirCushion_disabling_OK:
   IF NOT AirCushion_disable_request THEN
       Elevator_disable_request := TRUE;

       state := Elevator_disabling_OK;
   END_IF;

Elevator_disabling_OK:
   IF NOT Elevator_disable_request THEN
 	Testing_handler.Disable:=FALSE;

       state := Testing_ready_to_enable;
   END_IF;
END_CASE;

END_IF


(*** GENERIC DEVICES ***)

Elevator.DeviceOperation := OperationType;
Elevator.DeviceClock := TRUE;
Elevator.DeviceDiagnosticsEnabled := TRUE;
Elevator.DeviceEnablePreset := FALSE;
Elevator.DeviceEnabledSensor := Elevator_enabled;
Elevator.DeviceDisabledSensor := Elevator_disabled;
Elevator.DeviceEnableTime := Elevator_EnableTime;
Elevator.DeviceDisableTime := Elevator_DisableTime;
Elevator.DeviceType := DEVICE_WITH_DOUBLE_FEEDBACK OR DEVICE_WITH_DOUBLE_ACTUATION;
Elevator(DeviceEnableRequest := Elevator_enable_request, DeviceDisableRequest := Elevator_disable_request );
enable_Elevator := Elevator.EnableDevice;
Elevator_not_initialized:=Elevator.DeviceNotInitialized;
disable_Elevator := Elevator.DisableDevice;
Elevator_ActuatorFault := Elevator.DeviceActuatorFault;
Elevator_EnabledSensorFault := Elevator.DeviceEnabledSensorFault;
Elevator_DisabledSensorFault := Elevator.DeviceDisabledSensorFault;
Elevator_fault := Elevator.DeviceFault;

ExtractionCylinder.DeviceOperation := OperationType;
ExtractionCylinder.DeviceClock := TRUE;
ExtractionCylinder.DeviceDiagnosticsEnabled := TRUE;
ExtractionCylinder.DeviceEnablePreset := FALSE;
ExtractionCylinder.DeviceDisabledSensor := ExtractionCylinder_disabled;
ExtractionCylinder.DeviceEnableTime := ExtractionCylinder_EnableTime;
ExtractionCylinder.DeviceDisableTime := ExtractionCylinder_DisableTime;
ExtractionCylinder.DeviceType := DEVICE_WITH_DISABLE_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
ExtractionCylinder(DeviceEnableRequest := ExtractionCylinder_enable_request, DeviceDisableRequest := ExtractionCylinder_disable_request );
enable_ExtractionCylinder := ExtractionCylinder.EnableDevice;
ExtractionCylinder_not_initialized:=ExtractionCylinder.DeviceNotInitialized;
ExtractionCylinder_ActuatorFault := ExtractionCylinder.DeviceActuatorFault;
ExtractionCylinder_DisabledSensorFault := ExtractionCylinder.DeviceDisabledSensorFault;
ExtractionCylinder_fault := ExtractionCylinder.DeviceFault;

AirCushion.DeviceOperation := OperationType;
AirCushion.DeviceClock := TRUE;
AirCushion.DeviceDiagnosticsEnabled := TRUE;
AirCushion.DeviceEnablePreset := FALSE;
AirCushion.DeviceEnableTime := AirCushion_EnableTime;
AirCushion.DeviceDisableTime := AirCushion_DisableTime;
AirCushion.DeviceType := DEVICE_WITHOUT_FEEDBACK OR DEVICE_WITH_SINGLE_ACTUATION;
AirCushion(DeviceEnableRequest := AirCushion_enable_request, DeviceDisableRequest := AirCushion_disable_request );
enable_AirCushion := AirCushion.EnableDevice;
AirCushion_not_initialized:=AirCushion.DeviceNotInitialized;
AirCushion_ActuatorFault := AirCushion.DeviceActuatorFault;                A   , o � ]�           Fault_Detection ��d
    @    ��d�  K   �                                                                                                       
    @        	  K�*h  ��� � � ���                                            @                           ���        @	                       @                                                                                                          
    @         K 1   ���     ���                                            @                          ���        @	                       @                                                                                                          
    @        <" }A �1    ���     ���      ��                                    Sensors faults diagnosis @                          ���    	   Arial @                       @                                                                                                           
    @        �" �A �1     ��     �                                               @                          ���        @	                       @                                                                                                           
    @        	 ��5�  ��� �   ���                                            @                          ���        @	                       @                                                                                                          
    @        � ���  ���     ���                                            @                      N    ���        @	                       @                                                                                                          
    @        �X�   ���     ���      ��                                    Actuators faults diagnosis @                      +    ���    	   Arial @                       @                                                                                                           
    @        u���    ��     �                                               @                      ,    ���        @	                       @                                                                                                         
    @        ���9�    @                 #   Back to Plant @���     ���             @    �   ���    	   Arial @                      @    FESTO_Interface  �                                                                                                       
    @         V P�� }  ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        _ j  � t    ���     ���      ��                                    Distribution @                      �   ���    	   Arial @                       @                                                                                                           
    @        # � V � < �   ���     ��                                     EmptyWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        � F� ,�   ���     ��                                     EmptyWarehouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        U � � � �   ���     ��                                  5   EmptyWarehouseBlockedHigh OR EmptyWarehouseBlockedLow       EmptyWarehouse @                      �   ���        @	                       @                                                                                                           
    @        U � � � �   ���     ��                                  k   CylinderExtractionLoadInExtensivePositionBlockedHigh OR CylinderExtractionLoadInExtensivePositionBlockedLow    /   CylinderExtractionLoadIn
ExtensivePosition @                      �   ���        @	                       @                                                                                                           
    @        U � � �   ���     ��                                  o   CylinderExtractionLoadInRetroactivePositionBlockedHigh OR CylinderExtractionLoadInRetroactivePositionBlockedLow    1   CylinderExtractionLoadIn
RetroactivePosition @                      �   ���        @	                       @                                                                                                           
    @        U 
)�   ���     ��                                  U   RotaryMakerInPositionWarehouseBlockedHigh OR RotaryMakerInPositionWarehouseBlockedLow    "   RotaryMakerInPositionWarehouse @                      �   ���        @	                       @                                                                                                           
    @        U 2Q� A  ���     ��                                  [   RotaryMakerInPositionVerificationBlockedHigh OR RotaryMakerInPositionVerificationBlockedLow    %   RotaryMakerInPositionVerification @                      �   ���        @	                       @                                                                                                           
    @        U Zy� i  ���     ��                                  ;   VacuumGeneratorOkBlockedHigh OR VacuumGeneratorOkBlockedLow       VacuumGeneratorOk @                      �   ���        @	                       @                                                                                                           
    @        U ��� �  ���     ��                                  I   ReadyLoadForVerificationBlockedHigh OR ReadyLoadForVerificationBlockedLow       ReadyLoadForVerification @                      �   ���        @	                       @                                                                                                           
    @        U ��� �  ���     ��                                  ;   ColourMeasurementBlockedHigh OR ColourMeasurementBlockedLow       ColourMeasurement @                      �   ���        @	                       @                                                                                                           
    @        U ��� �  ���     ��                                  K   CylinderDownToMeasureLoadBlockedHigh OR CylinderDownToMeasureLoadBlockedLow       CylinderDownToMeasureLoad @                      �   ���        @	                       @                                                                                                           
    @        U �� 	  ���     ��                                  G   CylinderUpToMeasureLoadBlockedHigh OR CylinderUpToMeasureLoadBlockedLow       CylinderUpToMeasureLoad @                      �   ���        @	                       @                                                                                                           
    @        U "A� 1  ���     ��                                  9   VerificationBusyBlockedHigh OR VerificationBusyBlockedLow       VerificationBusy @                      �   ���        @	                       @                                                                                                           
    @        U Ji� Y  ���     ��                                  k   CylinderOfExtractionInRetroactivePositionBlockedHigh OR CylinderOfExtractionInRetroactivePositionBlockedLow    /   CylinderOfExtractionIn
RetroactivePosition @                      �   ���        @	                       @                                                                                                           
    @        U r�� �  ���     ��                                             MeasurementNotOk @                      �   ���        @	                       @                                                                                                           
    @        # � V � < �   ���     ��                                  4   CylinderExtractionLoadInExtensivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        � F� ,�   ���     ��                                  3   CylinderExtractionLoadInExtensivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # � V < �   ���     ��                                  6   CylinderExtractionLoadInRetroactivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        � F,�   ���     ��                                  5   CylinderExtractionLoadInRetroactivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # 
V )<   ���     ��                                  )   RotaryMakerInPositionWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        
F),  ���     ��                                  (   RotaryMakerInPositionWarehouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # 2V Q< A  ���     ��                                  ,   RotaryMakerInPositionVerificationBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        2FQ,A  ���     ��                                  +   RotaryMakerInPositionVerificationBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # ZV y< i  ���     ��                                     VacuumGeneratorOkBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ZFy,i  ���     ��                                     VacuumGeneratorOkBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # �V �< �  ���     ��                                  #   ReadyLoadForVerificationBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �F�,�  ���     ��                                  "   ReadyLoadForVerificationBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # �V �< �  ���     ��                                     ColourMeasurementBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �F�,�  ���     ��                                     ColourMeasurementBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # �V �< �  ���     ��                                  $   CylinderDownToMeasureLoadBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �F�,�  ���     ��                                  #   CylinderDownToMeasureLoadBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # �V < 	  ���     ��                                  "   CylinderUpToMeasureLoadBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �F,	  ���     ��                                  !   CylinderUpToMeasureLoadBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # "V A< 1  ���     ��                                     VerificationBusyBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        "FA,1  ���     ��                                     VerificationBusyBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # JV i< Y  ���     ��                                  4   CylinderOfExtractionInRetroactivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        JFi,Y  ���     ��                                  3   CylinderOfExtractionInRetroactivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        # rV �< �  ���     ��                                     MeasurementNotOkBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        rF�,�  ���     ��                                     MeasurementNotOkBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @         *K��   ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        S @U� J   ���     ���      ��                                    Distribution @                      �   ���    	   Arial @                       @                                                                                                           
    @        S h�� w  ���     ��                                  X   CylinderExtractsLoadFromWarehouseBlocked OR CylinderExtractsLoadFromWarehouseBlockedHigh    '   CylinderExtractsLoad
FromWarehouse @                      �   ���        @	                       @                                                                                                           
    @        S ��� �  ���     ��                                  H   RotaryMakerVsVerificationBlocked OR RotaryMakerVsVerificationBlockedHigh       RotaryMakerVsVerification @                      �   ���        @	                       @                                                                                                           
    @        S ��� �  ���     ��                                  4   VacuumGeneratorBlocked OR VacuumGeneratorBlockedHigh       VacuumGenerator @                      �   ���        @	                       @                                                                                                           
    @        S '�   ���     ��                                  :   ExpulsionAirVacuumBlocked OR ExpulsionAirVacuumBlockedHigh       ExpulsionAirVacuum @                      �   ���        @	                       @                                                                                                           
    @        S 0O� ?  ���     ��                                  L   ToLiftCylinderToMeasureLoadBlocked OR ToLiftCylinderToMeasureLoadBlockedHigh       ToLiftCylinderToMeasureLoad @                      �   ���        @	                       @                                                                                                           
    @        S Xw� g  ���     ��                                  N   ToLowerCylinderToMeasureLoadBlocked OR ToLowerCylinderToMeasureLoadBlockedHigh        ToLowerCylinderToMeasureLoad @                      �   ���        @	                       @                                                                                                           
    @        S ��� �  ���     ��                                  *   AirCushionBlocked OR AirCushionBlockedHigh       AirCushion @                      �   ���        @	                       @                                                                                                           
    @        S ��� �  ���     ��                                  \   ToExtendCylinderOfExtractionVsGuideBlocked OR ToExtendCylinderOfExtractionVsGuideBlockedHigh    )   ToExtendCylinderOfExtraction
VsGuide @                      �   ���        @	                       @                                                                                                           
    @        hD�*w  ���     ��                                  ,   CylinderExtractsLoadFromWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �D�*�  ���     ��                                  !   RotaryMakerVsWarehouseBlockedHigh       Blocked
High @                          ���        @	                       @                                                                                                           
    @        �D�*�  ���     ��                                  $   RotaryMakerVsVerificationBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        �D�*�  ���     ��                                     VacuumGeneratorBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        D'*  ���     ��                                     ExpulsionAirVacuumBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        0DO*?  ���     ��                                  &   ToLiftCylinderToMeasureLoadBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        XDw*g  ���     ��                                  '   ToLowerCylinderToMeasureLoadBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        �D�*�  ���     ��                                  .   ToExtendCylinderOfExtractionVsGuideBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        �D�*�  ���     ��                                     AirCushionBlockedHigh       Blocked
High @                         ���        @	                       @                                                                                                           
    @        ! hT �: w  ���     ��                                  (   CylinderExtractsLoadFromWarehouseBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! �T �: �  ���     ��                                     RotaryMakerVsWarehouseBlocked       Blocked @                      	   ���        @	                       @                                                                                                           
    @        ! �T �: �  ���     ��                                      RotaryMakerVsVerificationBlocked       Blocked @                      
   ���        @	                       @                                                                                                           
    @        ! �T �: �  ���     ��                                     VacuumGeneratorBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! T ':   ���     ��                                     ExpulsionAirVacuumBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! 0T O: ?  ���     ��                                  "   ToLiftCylinderToMeasureLoadBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! XT w: g  ���     ��                                  #   ToLowerCylinderToMeasureLoadBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! �T �: �  ���     ��                                  *   ToExtendCylinderOfExtractionVsGuideBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        ! �T �: �  ���     ��                                     AirCushionBlocked       Blocked @                         ���        @	                       @                                                                                                           
    @        S ��� �  ���     ��                                  B   RotaryMakerVsWarehouseBlocked OR RotaryMakerVsWarehouseBlockedHigh       RotaryMakerVsWarehouse @                         ���        @	                       @                                                                                                           
    @        WW ��  ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        �c mx m    ���     ���      ��                                    Processing @                      �   ���    	   Arial @                       @                                                                                                           
    @        �W 9���  ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        !c �x �m    ���     ���      ��                                    Assembly @                      �   ���    	   Arial @                       @                                                                                                           
    @        �� �� �   ���     ��                                          )   AlignementRotaryTableWithPositionings @                      �   ���        @	                       @                                                                                                           
    @        �� �� �   ���     ���                                         "   AvailableLoadForWorkingStation @                      �   ���        @	                       @                                                                                                           
    @        �� �� �   ���     ��                                          %   AvailableLoadInControlPositioning @                      �   ���        @	                       @                                                                                                           
    @        ��!  ���     ���                                         &   AvailableLoadInDrillingPositioning @                      �   ���        @	                       @                                                                                                           
    @        �+�J:  ���     ��                                          +   InControlLoadInWrongPositionToBeDrilled @                      �   ���        @	                       @                                                                                                           
    @        �S�rb  ���     ���                                            DrillingUnitUp @                      �   ���        @	                       @                                                                                                           
    @        �{���  ���     ��                                             DrillingUnitDown @                      �   ���        @	                       @                                                                                                           
    @        � �� ��   ���     ��                                             AvailableLoadForRobot @                      �   ���        @	                       @                                                                                                           
    @        � �� ��   ���     ���                                            RobotInInitialPosition @                      �   ���        @	                       @                                                                                                           
    @        � �� ��   ���     ��                                             RobotInAssemblyUnit @                      �   ���        @	                       @                                                                                                           
    @        �!�  ���     ���                                            RobotInPistonWarehouse @                      �   ���        @	                       @                                                                                                           
    @        +�J�:  ���     ��                                             RobotInSpringWarehouse @                      �   ���        @	                       @                                                                                                           
    @        S�r�b  ���     ���                                            RobotInCoverWarehouse @                      �   ���        @	                       @                                                                                                           
    @        {����  ���     ��                                          &   EmptyCoverHouse
InAssemblyStation @                      �   ���        @	                       @                                                                                                           
    @        �����  ���     ��                                          :   ToExtractCoverInAssemblyStationIn
RetroactivePosition @                      �   ���        @	                       @                                                                                                           
    @        �����  ���     ���                                         8   ToExtractCoverInAssemblyStationIn
ExtensivePosition @                      �   ���        @	                       @                                                                                                           
    @        ���  ���     ��                                             PistonSelectorIsOnTheRight @                      �   ���        @	                       @                                                                                                           
    @        �<�,  ���     ���                                            PistonSelectorIsOnTheLeft @                      �   ���        @	                       @                                                                                                           
    @        F�e�U  ���     ��                                          9   ToExtractSpringInAssemblyStationIn
ExtensivePosition @                      �   ���        @	                       @                                                                                                           
    @        n���}  ���     ���                                         ;   ToExtractSpringInAssemblyStationIn
RetroactivePosition @                      �   ���        @	                       @                                                                                                           
    @        �� .� �   ���     ��                                     AvailableLoadForRobotBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �� .� �   ���     ��                                      RobotInInitialPositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �� /� �   ���     ��                                     RobotInAssemblyUnitBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �/!  ���     ��                                      RobotInPistonWarehouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �+/J:  ���     ��                                      RobotInSpringWarehouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �S/rb  ���     ��                                     RobotInCoverWarehouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �{/��  ���     ��                                  *   EmptyCoverHouseInAssemblyStationBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ��/��  ���     ��                                  >   ToExtractCoverInAssemblyStationInRetroactivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ��/��  ���     ��                                  <   ToExtractCoverInAssemblyStationInExtensivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ��0  ���     ��                                  $   PistonSelectorIsOnTheRightBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �0<,  ���     ��                                  #   PistonSelectorIsOnTheLeftBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �F0eU  ���     ��                                  =   ToExtractSpringInAssemblyStationInExtensivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �n0�}  ���     ��                                  ?   ToExtractSpringInAssemblyStationInRetroactivePositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �� � ��   ���     ��                                      AvailableLoadForRobotBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �� � ��   ���     ��                                  !   RobotInInitialPositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �� � ��   ���     ��                                     RobotInAssemblyUnitBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �	!�  ���     ��                                  !   RobotInPistonWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �+J�:  ���     ��                                  !   RobotInSpringWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �S	r�b  ���     ��                                      RobotInCoverWarehouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �{���  ���     ��                                  +   EmptyCoverHouseInAssemblyStationBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �����  ���     ��                                  ?   ToExtractCoverInAssemblyStationInRetroactivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ��	���  ���     ��                                  =   ToExtractCoverInAssemblyStationInExtensivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ��	�  ���     ��                                  %   PistonSelectorIsOnTheRightBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �
<�,  ���     ��                                  $   PistonSelectorIsOnTheLeftBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �F	e�U  ���     ��                                  >   ToExtractSpringInAssemblyStationInExtensivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �n
��}  ���     ��                                  @   ToExtractSpringInAssemblyStationInRetroactivePositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �� �� ��   ���     ��                                  /   AlignementRotaryTableWithPositioningsBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �� �� ��   ���     ��                                  (   AvailableLoadForWorkingStationBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �� �� ��   ���     ��                                  +   AvailableLoadInControlPositioningBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ��!�  ���     ��                                  ,   AvailableLoadInDrillingPositioningBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �+�J�:  ���     ��                                  1   InControlLoadInWrongPositionToBeDrilledBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �S�r�b  ���     ��                                     DrillingUnitUpBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �{����  ���     ��                                     DrillingUnitDownBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ]� �� v�   ���     ��                                  0   AlignementRotaryTableWithPositioningsBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ^� �� w�   ���     ��                                  )   AvailableLoadForWorkingStationBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ^� �� w�   ���     ��                                  ,   AvailableLoadInControlPositioningBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        _�!x  ���     ��                                  -   AvailableLoadInDrillingPositioningBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ^+�Jw:  ���     ��                                  2   InControlLoadInWrongPositionToBeDrilledBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        _S�rxb  ���     ��                                     DrillingUnitUpBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ^{��w�  ���     ��                                     DrillingUnitDownBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �����  ���     ���                                            RobotProgramRunning @                      �   ���        @	                       @                                                                                                           
    @        ��1��  ���     ��                                     RobotProgramRunningBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        Y,q���  ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        �5JJ�?   ���     ���      ��                                    Processing @                      �   ���    	   Arial @                       @                                                                                                           
    @        z,� 
&  ���     ���                                             @                      �   ���        @	                       @                                                                                                           
    @        �5mJ?   ���     ���      ��                                    Assembly @                      �   ���    	   Arial @                       @                                                                                                           
    @        �ia�x  ���     ��                                          &   ToExtractSpringIn
AssemblyStation @                      �   ���        @	                       @                                                                                                           
    @        ��a��  ���     ���                                            PistonSelectorGoOnTheRight @                      �   ���        @	                       @                                                                                                           
    @        ��a�	�  ���     ���                                            PistonSelectorGoOnTheLeft @                      �   ���        @	                       @                                                                                                           
    @        ��a �  ���     ���                                         ,   ToExtractCoverInAssembly
StationForward @                      �   ���        @	                       @                                                                                                           
    @        �	a(	  ���     ���                                         .   BlockingCylinderForwardIn
AssemblyStation @                      �   ���        @	                       @                                                                                                           
    @        �0eO
?  ���     ���                                            RobotGoToInitialPosition @                      �   ���        @	                       @                                                                                                           
    @        �f�
�  ���     ���                                            RobotGoToSpringWarehouse @                      �   ���        @	                       @                                                                                                           
    @        �Wfvf  ���     ���                                            RobotGoToPistonWarehouse @                      �   ���        @	                       @                                                                                                           
    @        �i8��x  ���     ��                                             RotaryTableMotor @                      �   ���        @	                       @                                                                                                           
    @        ��8���  ���     ���                                         "   ToLowerCylinder
ToInspectLoad @                      �   ���        @	                       @                                                                                                           
    @        ��8���  ���     ���                                            DrillingUnitActive @                      �   ���        @	                       @                                                                                                           
    @        ��8 ��  ���     ���                                            ToLowerDrillingUnit @                      �   ���        @	                       @                                                                                                           
    @        �	8(�  ���     ���                                            ToLiftDrillingUnit @                      �   ���        @	                       @                                                                                                           
    @        �Y8x�h  ���     ���                                            ExpellingLeverActive @                      �   ���        @	                       @                                                                                                           
    @        �18P�@  ���     ���                                         /   BlockingCylinderForwardIn
DrillingPosition @                      �   ���        @	                       @                                                                                                           
    @        ai��zx  ���     ��                                     RotaryTableMotorBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        a���z�  ���     ��                                  '   ToLowerCylinderToInspectLoadBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        a���z�  ���     ��                                     DrillingUnitActiveBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        a�� z�  ���     ��                                     ToLowerDrillingUnitBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        a	�(z  ���     ��                                     ToLiftDrillingUnitBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        6ii�Ox  ���     ��                                     RotaryTableMotorBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        6�i�O�  ���     ��                                  &   ToLowerCylinderToInspectLoadBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        6�i�O�  ���     ��                                     DrillingUnitActiveBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        6�i O�  ���     ��                                     ToLowerDrillingUnitBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        6	i(O  ���     ��                                     ToLiftDrillingUnitBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �i���x  ���     ��                                  +   ToExtractSpringInAssemblyStationBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ������  ���     ��                                  %   PistonSelectorGoOnTheRightBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ������  ���     ��                                  $   PistonSelectorGoOnTheLeftBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ��� ��  ���     ��                                  1   ToExtractCoverInAssemblyStationForwardBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ^i��wx  ���     ��                                  *   ToExtractSpringInAssemblyStationBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^���w�  ���     ��                                  $   PistonSelectorGoOnTheRightBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^���w�  ���     ��                                  #   PistonSelectorGoOnTheLeftBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^�� w�  ���     ��                                  0   ToExtractCoverInAssemblyStationForwardBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^0�Ow?  ���     ��                                  "   RobotGoToInitialPositionBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^W�vwf  ���     ��                                     RobotGoToPistonHouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ^��w�  ���     ��                                     RobotGoToSpringHouseBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        �0�O�?  ���     ��                                  #   RobotGoToInitialPositionBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �W�v�f  ���     ��                                     RobotGoToPistonHouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        �����  ���     ��                                     RobotGoToSpringHouseBlockedHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ��f  ���     ���                                            RobotEngine @                      �   ���        @	                       @                                                                                                           
    @        ^��w  ���     ��                                     RobotEngineBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ��g��  ���     ���                                            RobotGoToCoverWarehouse @                      �   ���        @	                       @                                                                                                           
    @        ��g��  ���     ���                                         $   RobotTakeCurrentLoad
ToAssembly @                      �   ���        @	                       @                                                                                                           
    @        _���x�  ���     ��                                     RobotGoToCoverHouseBlockLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        _���x�  ���     ��                                  (   RobotTakeCurrentLoadToAssemblyBlockedLow       Blocked
Low @                      �   ���        @	                       @                                                                                                           
    @        ������  ���     ��                                     RobotGoToCoverHouseBlockHigh       Blocked
High @                      �   ���        @	                       @                                                                                                           
    @        ������  ���     ��                                  )   RobotTakeCurrentLoadToAssemblyBlockedHigh       Blocked
High @                      �   ���        @	                       @             �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                  @   , �� h�           Fault_Simulation ��d
    @    ��d  2   �                                                                                                       
    @          
 /�o  ��� � � ���                                            @                           ���        @                       @                                                                                                          
    @        Y �K 1   ���     ���                                            @                      %    ���        @                       @                                                                                                          
    @        m" �A 1    ���     ���      ��                                    Sensors faults simulation @                          ���    	   Arial @                       @                                                                                                           
    @        
 ��H�  ��� �   ���                                            @                      �    ���        @                       @                                                                                                          
    @        6��!�  ���     ���                                            @                      �    ���        @                       @                                                                                                          
    @        L���   ���     ���      ��                                    Actuators faults simulation @                      �    ���    	   Arial @                       @                                                                                                         
    @        ���?     @                 #   Back to Plant @���     ���             @    �   ���    	   Arial @                      @    FESTO_Interface  �                                                                                                       
    @         Z K�� �  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        ( r 7� � |    ���     ���      ��                                 !   Distribution and Verification @                      �   ���    	   Arial @                       @                                                                                                           
    @        P � � � �   ���     ���                                 1   EmptyWarehouseBlockHigh OR EmptyWarehouseBlockLow       EmptyWarehouse @                      �   ���        @                       @                                                                                                           
    @        P � � � �   ���     ���                                 g   CylinderExtractionLoadInExtensivePositionBlockHigh OR CylinderExtractionLoadInExtensivePositionBlockLow    /   CylinderExtractionLoadIn
ExtensivePosition @                      �   ���        @                       @                                                                                                           
    @        P � 	� �   ���     ���                                 k   CylinderExtractionLoadInRetroactivePositionBlockHigh OR CylinderExtractionLoadInRetroactivePositionBlockLow    1   CylinderExtractionLoadIn
RetroactivePosition @                      �   ���        @                       @                                                                                                           
    @        P 1� !  ���     ���                                 Q   RotaryMakerInPositionWarehouseBlockHigh OR RotaryMakerInPositionWarehouseBlockLow    "   RotaryMakerInPositionWarehouse @                      �   ���        @                       @                                                                                                           
    @        P :Y� I  ���     ���                                 W   RotaryMakerInPositionVerificationBlockHigh OR RotaryMakerInPositionVerificationBlockLow    %   RotaryMakerInPositionVerification @                      �   ���        @                       @                                                                                                           
    @        P b�� q  ���     ���                                 7   VacuumGeneratorOkBlockHigh OR VacuumGeneratorOkBlockLow       VacuumGeneratorOk @                      �   ���        @                       @                                                                                                           
    @        P ��� �  ���     ���                                 E   ReadyLoadForVerificationBlockHigh OR ReadyLoadForVerificationBlockLow       ReadyLoadForVerification @                      �   ���        @                       @                                                                                                           
    @        P ��� �  ���     ���                                 7   ColourMeasurementBlockHigh OR ColourMeasurementBlockLow       ColourMeasurement @                      �   ���        @                       @                                                                                                           
    @        P ��� �  ���     ���                                 G   CylinderDownToMeasureLoadBlockHigh OR CylinderDownToMeasureLoadBlockLow       CylinderDownToMeasureLoad @                      �   ���        @                       @                                                                                                           
    @        P !�   ���     ���                                 C   CylinderUpToMeasureLoadBlockHigh OR CylinderUpToMeasureLoadBlockLow       CylinderUpToMeasureLoad @                      �   ���        @                       @                                                                                                           
    @        P *I� 9  ���     ���                                 5   VerificationBusyBlockHigh OR VerificationBusyBlockLow       VerificationBusy @                      �   ���        @                       @                                                                                                           
    @        P Rq� a  ���     ���                                 g   CylinderOfExtractionInRetroactivePositionBlockHigh OR CylinderOfExtractionInRetroactivePositionBlockLow    /   CylinderOfExtractionIn
RetroactivePosition @                      �   ���        @                       @                                                                                                           
    @        P z�� �  ���     ���                                 5   MeasurementNotOkBlockHigh OR MeasurementNotOkBlockLow       MeasurementNotOk @                      �   ���        @                       @                                                                                                         
    @         � Q � 7 �     @                 !   Block
HIGH @���     ���             @    �   ���        @    EmptyWarehouseBlockHigh                 @       �                                                                                                     
    @        � A� '�     @                     Block
LOW @���     ���             @    �   ���        @    EmptyWarehouseBlockLow                 @       �                                                                                                     
    @         � Q � 7 �     @                 !   Block
HIGH @���     ���             @    �   ���        @ 2   CylinderExtractionLoadInExtensivePositionBlockHigh                 @       �                                                                                                     
    @        � A� '�     @                     Block
LOW @���     ���             @    �   ���        @ 1   CylinderExtractionLoadInExtensivePositionBlockLow                 @       �                                                                                                     
    @         � Q 	7 �     @                 !   Block
HIGH @���     ���             @    �   ���        @ 4   CylinderExtractionLoadInRetroactivePositionBlockHigh                 @       �                                                                                                     
    @        � A	'�     @                     Block
LOW @���     ���             @    �   ���        @ 3   CylinderExtractionLoadInRetroactivePositionBlockLow                 @       �                                                                                                     
    @         Q 17 !    @                 !   Block
HIGH @���     ���             @    �   ���        @ '   RotaryMakerInPositionWarehouseBlockHigh                 @       �                                                                                                     
    @        A1'!    @                     Block
LOW @���     ���             @    �   ���        @ &   RotaryMakerInPositionWarehouseBlockLow                 @       �                                                                                                     
    @         :Q Y7 I    @                 !   Block
HIGH @���     ���             @    �   ���        @ *   RotaryMakerInPositionVerificationBlockHigh                 @       �                                                                                                     
    @        :AY'I    @                     Block
LOW @���     ���             @    �   ���        @ )   RotaryMakerInPositionVerificationBlockLow                 @       �                                                                                                     
    @         bQ �7 q    @                 !   Block
HIGH @���     ���             @    �   ���        @    VacuumGeneratorOkBlockHigh                 @       �                                                                                                     
    @        bA�'q    @                     Block
LOW @���     ���             @    �   ���        @    VacuumGeneratorOkBlockLow                 @       �                                                                                                     
    @         �Q �7 �    @                 !   Block
HIGH @���     ���             @    �   ���        @ !   ReadyLoadForVerificationBlockHigh                 @       �                                                                                                     
    @        �A�'�    @                     Block
LOW @���     ���             @    �   ���        @     ReadyLoadForVerificationBlockLow                 @       �                                                                                                     
    @         �Q �7 �    @                 !   Block
HIGH @���     ���             @    �   ���        @    ColourMeasurementBlockHigh                 @       �                                                                                                     
    @        �A�'�    @                     Block
LOW @���     ���             @    �   ���        @    ColourMeasurementBlockLow                 @       �                                                                                                     
    @         �Q �7 �    @                 !   Block
HIGH @���     ���             @    �   ���        @ "   CylinderDownToMeasureLoadBlockHigh                 @       �                                                                                                     
    @        �A�'�    @                     Block
LOW @���     ���             @    �   ���        @ !   CylinderDownToMeasureLoadBlockLow                 @       �                                                                                                     
    @         Q !7     @                 !   Block
HIGH @���     ���             @    �   ���        @     CylinderUpToMeasureLoadBlockHigh                 @       �                                                                                                     
    @        A!'    @                     Block
LOW @���     ���             @    �   ���        @    CylinderUpToMeasureLoadBlockLow                 @       �                                                                                                     
    @         *Q I7 9    @                 !   Block
HIGH @���     ���             @    �   ���        @    VerificationBusyBlockHigh                 @       �                                                                                                     
    @        *AI'9    @                     Block
LOW @���     ���             @    �   ���        @    VerificationBusyBlockLow                 @       �                                                                                                     
    @         RQ q7 a    @                 !   Block
HIGH @���     ���             @    �   ���        @ 2   CylinderOfExtractionInRetroactivePositionBlockHigh                 @       �                                                                                                     
    @        RAq'a    @                     Block
LOW @���     ���             @    �   ���        @ 1   CylinderOfExtractionInRetroactivePositionBlockLow                 @       �                                                                                                     
    @         zQ �7 �    @                 !   Block
HIGH @���     ���             @    �   ���        @    MeasurementNotOkBlockHigh                 @       �                                                                                                     
    @        zA�'�    @                     Block
LOW @���     ���             @    �   ���        @    MeasurementNotOkBlockLow                 @       �                                                                                                       
    @         ;a��   ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        ( OKd� Y   ���     ���      ��                                 !   Distribution and Verification @                      �   ���    	   Arial @                       @                                                                                                           
    @        \ w�� �  ���     ���                                 T   CylinderExtractsLoadFromWarehouseBlock OR CylinderExtractsLoadFromWarehouseBlockHigh    '   CylinderExtractsLoad
FromWarehouse @                      �   ���        @                       @                                                                                                           
    @        \ ��� �  ���     ���                                 >   RotaryMakerVsWarehouseBlock OR RotaryMakerVsWarehouseBlockHigh       RotaryMakerVsWarehouse @                      �   ���        @                       @                                                                                                           
    @        \ ��� �  ���     ���                                 D   RotaryMakerVsVerificationBlock OR RotaryMakerVsVerificationBlockHigh       RotaryMakerVsVerification @                          ���        @                       @                                                                                                           
    @        \ �� �  ���     ���                                 0   VacuumGeneratorBlock OR VacuumGeneratorBlockHigh       VacuumGenerator @                         ���        @                       @                                                                                                           
    @        \ 6� &  ���     ���                                 6   ExpulsionAirVacuumBlock OR ExpulsionAirVacuumBlockHigh       ExpulsionAirVacuum @                         ���        @                       @                                                                                                           
    @        \ ?^� N  ���     ���                                 H   ToLiftCylinderToMeasureLoadBlock OR ToLiftCylinderToMeasureLoadBlockHigh       ToLiftCylinderToMeasureLoad @                         ���        @                       @                                                                                                           
    @        \ g�� v  ���     ���                                 J   ToLowerCylinderToMeasureLoadBlock OR ToLowerCylinderToMeasureLoadBlockHigh        ToLowerCylinderToMeasureLoad @                         ���        @                       @                                                                                                           
    @        \ ��� �  ���     ���                                 &   AirCushionBlock OR AirCushionBlockHigh       AirCushion @                         ���        @                       @                                                                                                           
    @        \ ��� �  ���     ���                                 X   ToExtendCylinderOfExtractionVsGuideBlock OR ToExtendCylinderOfExtractionVsGuideBlockHigh    )   ToExtendCylinderOfExtraction
VsGuide @                         ���        @                       @                                                                                                         
    @        wM�3�    @                 !   Block
HIGH @���     ���             @       ���        @ *   CylinderExtractsLoadFromWarehouseBlockHigh                 @       �                                                                                                     
    @        �M�3�    @                 !   Block
HIGH @���     ���             @       ���        @    RotaryMakerVsWarehouseBlockHigh                 @       �                                                                                                     
    @        �M�3�    @                 !   Block
HIGH @���     ���             @    	   ���        @ "   RotaryMakerVsVerificationBlockHigh                 @       �                                                                                                     
    @        �M3�    @                 !   Block
HIGH @���     ���             @    
   ���        @    VacuumGeneratorBlockHigh                 @       �                                                                                                     
    @        M63&    @                 !   Block
HIGH @���     ���             @       ���        @    ExpulsionAirVacuumBlockHigh                 @       �                                                                                                     
    @        ?M^3N    @                 !   Block
HIGH @���     ���             @       ���        @ $   ToLiftCylinderToMeasureLoadBlockHigh                 @       �                                                                                                     
    @        gM�3v    @                 !   Block
HIGH @���     ���             @       ���        @ %   ToLowerCylinderToMeasureLoadBlockHigh                 @       �                                                                                                     
    @        �M�3�    @                 !   Block
HIGH @���     ���             @       ���        @ ,   ToExtendCylinderOfExtractionVsGuideBlockHigh                 @       �                                                                                                     
    @        �M�3�    @                 !   Block
HIGH @���     ���             @       ���        @    AirCushionBlockHigh                 @       �                                                                                                     
    @        * w] �C �    @                    Block @���     ���             @       ���        @ &   CylinderExtractsLoadFromWarehouseBlock                 @       �                                                                                                     
    @        * �] �C �    @                    Block @���     ���             @       ���        @    RotaryMakerVsWarehouseBlock                 @       �                                                                                                     
    @        * �] �C �    @                    Block @���     ���             @       ���        @    RotaryMakerVsVerificationBlock                 @       �                                                                                                     
    @        * �] C �    @                    Block @���     ���             @       ���        @    VacuumGeneratorBlock                 @       �                                                                                                     
    @        * ] 6C &    @                    Block @���     ���             @       ���        @    ExpulsionAirVacuumBlock                 @       �                                                                                                     
    @        * ?] ^C N    @                    Block @���     ���             @       ���        @     ToLiftCylinderToMeasureLoadBlock                 @       �                                                                                                     
    @        * g] �C v    @                    Block @���     ���             @       ���        @ !   ToLowerCylinderToMeasureLoadBlock                 @       �                                                                                                     
    @        * �] �C �    @                    Block @���     ���             @       ���        @ (   ToExtendCylinderOfExtractionVsGuideBlock                 @       �                                                                                                     
    @        * �] �C �    @                    Block @���     ���             @       ���        @    AirCushionBlock                 @       �                                                                                                       
    @        �Z "�t�  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        W_ ��	  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �q i� �    ���     ���      ��                                    Processing @                      �   ���    	   Arial @                       @                                                                                                           
    @        t �� s�    ���     ���      ��                                    Assembly @                      �   ���    	   Arial @                       @                                                                                                           
    @        �� � �   ���     ��                                          )   AlignementRotaryTableWithPositionings @                      �   ���        @                       @                                                                                                           
    @        �� � 	�   ���     ���                                         "   AvailableLoadForWorkingStation @                      �   ���        @                       @                                                                                                           
    @        �� �

  ���     ��                                          %   AvailableLoadInControlPositioning @                      �   ���        @                       @                                                                                                           
    @        �#�B	2  ���     ���                                         &   AvailableLoadInDrillingPositioning @                      �   ���        @                       @                                                                                                           
    @        �L�k	[  ���     ��                                          -   InControlLoadInWrongPosition
ToBeDrilled @                      �   ���        @                       @                                                                                                           
    @        �t��	�  ���     ���                                            DrillingUnitUp @                      �   ���        @                       @                                                                                                           
    @        ���	�  ���     ��                                             DrillingUnitDown @                      �   ���        @                       @                                                                                                         
    @        a� �� z�     @                 !   Block
HIGH @���     ���             @    �   ���        @ .   AlignementRotaryTableWithPositioningsBlockHigh                 @       �                                                                                                     
    @        a� �� z�     @                 !   Block
HIGH @���     ���             @    �   ���        @ '   AvailableLoadForWorkingStationBlockHigh                 @       �                                                                                                     
    @        a� �z
    @                 !   Block
HIGH @���     ���             @    �   ���        @ *   AvailableLoadInControlPositioningBlockHigh                 @       �                                                                                                     
    @        a"�Bz2    @                 !   Block
HIGH @���     ���             @    �   ���        @ +   AvailableLoadInDrillingPositioningBlockHigh                 @       �                                                                                                     
    @        aK�kz[    @                 !   Block
HIGH @���     ���             @    �   ���        @ 0   InControlLoadInWrongPositionToBeDrilledBlockHigh                 @       �                                                                                                     
    @        as��z�    @                 !   Block
HIGH @���     ���             @    �   ���        @    DrillingUnitUpBlockHigh                 @       �                                                                                                     
    @        }� �� ��     @                     Block
LOW @���     ���             @    �   ���        @ -   AlignementRotaryTableWithPositioningsBlockLow                 @       �                                                                                                     
    @        }� �� ��     @                     Block
LOW @���     ���             @    �   ���        @ &   AvailableLoadForWorkingStationBlockLow                 @       �                                                                                                     
    @        }� ��
    @                     Block
LOW @���     ���             @    �   ���        @ )   AvailableLoadInControlPositioningBlockLow                 @       �                                                                                                     
    @        }"�B�2    @                     Block
LOW @���     ���             @    �   ���        @ *   AvailableLoadInDrillingPositioningBlockLow                 @       �                                                                                                     
    @        }K�k�[    @                     Block
LOW @���     ���             @    �   ���        @ /   InControlLoadInWrongPositionToBeDrilledBlockLow                 @       �                                                                                                     
    @        }s����    @                     Block
LOW @���     ���             @    �   ���        @    DrillingUnitUpBlockLow                 @       �                                                                                                     
    @        b���{�    @                 !   Block
HIGH @���     ���             @    �   ���        @    DrillingUnitDownBlockHigh                 @       �                                                                                                     
    @        }�����    @                     Block
LOW @���     ���             @    �   ���        @    DrillingUnitDownBlockLow                 @       �                                                                                                       
    @        � �� v�   ���     ��                                             AvailableLoadForRobot @                      �   ���        @                       @                                                                                                           
    @        � �� v�   ���     ���                                            RobotInInitialPosition @                      �   ���        @                       @                                                                                                           
    @        � �u	  ���     ��                                             RobotInAssemblyUnit @                      �   ���        @                       @                                                                                                           
    @        "�Au1  ���     ���                                            RobotInPistonWarehouse @                      �   ���        @                       @                                                                                                           
    @        K�juZ  ���     ��                                             RobotInSpringWarehouse @                      �   ���        @                       @                                                                                                           
    @        s��u�  ���     ���                                            RobotInCoverWarehouse @                      �   ���        @                       @                                                                                                           
    @        ���u�  ���     ��                                          &   EmptyCoverHouse
InAssemblyStation @                      �   ���        @                       @                                                                                                           
    @        ���u�  ���     ��                                          :   ToExtractCoverInAssemblyStationIn
RetroactivePosition @                      �   ���        @                       @                                                                                                           
    @        ��u�  ���     ���                                         8   ToExtractCoverInAssemblyStationIn
ExtensivePosition @                      �   ���        @                       @                                                                                                           
    @        �4u$  ���     ��                                             PistonSelectorIsOnTheRight @                      �   ���        @                       @                                                                                                           
    @        =�\uL  ���     ���                                            PistonSelectorIsOnTheLeft @                      �   ���        @                       @                                                                                                           
    @        f��uu  ���     ��                                          9   ToExtractSpringInAssemblyStationIn
ExtensivePosition @                      �   ���        @                       @                                                                                                           
    @        ���u�  ���     ���                                         ;   ToExtractSpringInAssemblyStationIn
RetroactivePosition @                      �   ���        @                       @                                                                                                         
    @        �� � ��     @                 !   Block
HIGH @���     ���             @    �   ���        @    AvailableLoadForRobotBlockHigh                 @       �                                                                                                     
    @        �� � ��     @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotInInitialPositionBlockHigh                 @       �                                                                                                     
    @        �� �	    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotInAssemblyUnitBlockHigh                 @       �                                                                                                     
    @        �!A�1    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotInPistonWarehouseBlockHigh                 @       �                                                                                                     
    @        �Jj�Z    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotInSpringWarehouseBlockHigh                 @       �                                                                                                     
    @        �r���    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotInCoverWarehouseBlockHigh                 @       �                                                                                                     
    @        �����    @                 !   Block
HIGH @���     ���             @    �   ���        @ )   EmptyCoverHouseInAssemblyStationBlockHigh                 @       �                                                                                                     
    @        �����    @                 !   Block
HIGH @���     ���             @    �   ���        @ =   ToExtractCoverInAssemblyStationInRetroactivePositionBlockHigh                 @       �                                                                                                     
    @        ����    @                 !   Block
HIGH @���     ���             @    �   ���        @ ;   ToExtractCoverInAssemblyStationInExtensivePositionBlockHigh                 @       �                                                                                                     
    @        �4�$    @                 !   Block
HIGH @���     ���             @    �   ���        @ #   PistonSelectorIsOnTheRightBlockHigh                 @       �                                                                                                     
    @        �<\�L    @                 !   Block
HIGH @���     ���             @    �   ���        @ "   PistonSelectorIsOnTheLeftBlockHigh                 @       �                                                                                                     
    @        �e��u    @                 !   Block
HIGH @���     ���             @    �   ���        @ <   ToExtractSpringInAssemblyStationInExtensivePositionBlockHigh                 @       �                                                                                                     
    @        �����    @                 !   Block
HIGH @���     ���             @    �   ���        @ >   ToExtractSpringInAssemblyStationInRetroactivePositionBlockHigh                 @       �                                                                                                     
    @        �� � ��     @                     Block
LOW @���     ���             @    �   ���        @    AvailableLoadForRobotBlockLow                 @       �                                                                                                     
    @        �� � ��     @                     Block
LOW @���     ���             @    �   ���        @    RobotInInitialPositionBlockLow                 @       �                                                                                                     
    @        �� �	    @                     Block
LOW @���     ���             @    �   ���        @    RobotInAssemblyUnitBlockLow                 @       �                                                                                                     
    @        �!A�1    @                     Block
LOW @���     ���             @    �   ���        @    RobotInPistonWarehouseBlockLow                 @       �                                                                                                     
    @        �Jj�Z    @                     Block
LOW @���     ���             @    �   ���        @    RobotInSpringWarehouseBlockLow                 @       �                                                                                                     
    @        �r���    @                     Block
LOW @���     ���             @    �   ���        @    RobotInCoverWarehouseBlockLow                 @       �                                                                                                     
    @        �����    @                     Block
LOW @���     ���             @    �   ���        @ (   EmptyCoverHouseInAssemblyStationBlockLow                 @       �                                                                                                     
    @        �����    @                     Block
LOW @���     ���             @    �   ���        @ <   ToExtractCoverInAssemblyStationInRetroactivePositionBlockLow                 @       �                                                                                                     
    @        ����    @                     Block
LOW @���     ���             @    �   ���        @ :   ToExtractCoverInAssemblyStationInExtensivePositionBlockLow                 @       �                                                                                                     
    @        �4�$    @                     Block
LOW @���     ���             @    �   ���        @ "   PistonSelectorIsOnTheRightBlockLow                 @       �                                                                                                     
    @        �<\�L    @                     Block
LOW @���     ���             @    �   ���        @ !   PistonSelectorIsOnTheLeftBlockLow                 @       �                                                                                                     
    @        �e��u    @                     Block
LOW @���     ���             @    �   ���        @ ;   ToExtractSpringInAssemblyStationInExtensivePositionBlockLow                 @       �                                                                                                     
    @        �����    @                     Block
LOW @���     ���             @    �   ���        @ =   ToExtractSpringInAssemblyStationInRetroactivePositionBlockLow                 @       �                                                                                                       
    @        k;���  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �KW`�U   ���     ���      ��                                    Processing @                      �   ���    	   Arial @                       @                                                                                                           
    @        �;�*(2  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �J�_%T   ���     ���      ��                                    Assembly @                      �   ���    	   Arial @                       @                                                                                                           
    @        �r��*�  ���     ��                                          &   ToExtractSpringIn
AssemblyStation @                      �   ���        @                       @                                                                                                           
    @        ����+�  ���     ���                                            PistonSelectorGoOnTheRight @                      �   ���        @                       @                                                                                                           
    @        ����*�  ���     ���                                            PistonSelectorGoOnTheLeft @                      �   ���        @                       @                                                                                                           
    @        ���	+�  ���     ���                                         ,   ToExtractCoverInAssembly
StationForward @                      �   ���        @                       @                                                                                                           
    @        ��1*!  ���     ���                                         .   BlockingCylinderForwardIn
AssemblyStation @                      �   ���        @                       @                                                                                                           
    @        �9X(H  ���     ���                                            RobotGoToInitialPosition @                      �   ���        @                       @                                                                                                           
    @        ���'�  ���     ���                                            RobotGoToSpringHouse @                      �   ���        @                       @                                                                                                           
    @        �`(o  ���     ���                                            RobotGoToPistonHouse @                      �   ���        @                       @                                                                                                           
    @        �sN���  ���     ��                                             RotaryTableMotor @                      �   ���        @                       @                                                                                                           
    @        ��N���  ���     ���                                         "   ToLowerCylinder
ToInspectLoad @                      �   ���        @                       @                                                                                                           
    @        ��N���  ���     ���                                            DrillingUnitActive @                      �   ���        @                       @                                                                                                           
    @        ��N��  ���     ���                                            ToLowerDrillingUnit @                      �   ���        @                       @                                                                                                           
    @        �N3�#  ���     ���                                            ToLiftDrillingUnit @                      �   ���        @                       @                                                                                                           
    @        �dN��s  ���     ���                                            ExpellingLeverActive @                      �   ���        @                       @                                                                                                           
    @        �<N[�K  ���     ���                                         2   BlockingCylinderForwardIn
DrillingPositioning @                      �   ���        @                       @                                                                                                         
    @        Hr{�a�    @                     Block
LOW @���     ���             @    �   ���        @    RotaryTableMotorBlockLow                 @       �                                                                                                     
    @        H�{�a�    @                     Block
LOW @���     ���             @    �   ���        @ $   ToLowerCylinderToInspectLoadBlockLow                 @       �                                                                                                     
    @        H�{�a�    @                     Block
LOW @���     ���             @    �   ���        @    DrillingUnitActiveBlockLow                 @       �                                                                                                     
    @        H�{a�    @                     Block
LOW @���     ���             @    �   ���        @    ToLowerDrillingUnitBlockLow                 @       �                                                                                                     
    @        I|3b#    @                     Block
LOW @���     ���             @    �   ���        @    ToLiftDrillingUnitBlockLow                 @       �                                                                                                     
    @        H;{[aK    @                     Block
LOW @���     ���             @    �   ���        @ 4   BlockingCylinderForwardInDrillingPositioningBlockLow                 @       �                                                                                                     
    @        Hc{�as    @                     Block
LOW @���     ���             @    �   ���        @    ExpellingLeverActiveBlockLow                 @       �                                                                                                     
    @        wr����    @                 !   Block
HIGH @���     ���             @    �   ���        @    RotaryTableMotorBlockHigh                 @       �                                                                                                     
    @        w�����    @                 !   Block
HIGH @���     ���             @    �   ���        @ %   ToLowerCylinderToInspectLoadBlockHigh                 @       �                                                                                                     
    @        w�����    @                 !   Block
HIGH @���     ���             @    �   ���        @    DrillingUnitActiveBlockHigh                 @       �                                                                                                     
    @        w����    @                 !   Block
HIGH @���     ���             @    �   ���        @    ToLowerDrillingUnitBlockHigh                 @       �                                                                                                     
    @        x�3�#    @                 !   Block
HIGH @���     ���             @    �   ���        @    ToLiftDrillingUnitBlockHigh                 @       �                                                                                                     
    @        w;�[�K    @                 !   Block
HIGH @���     ���             @    �   ���        @ 5   BlockingCylinderForwardInDrillingPositioningBlockHigh                 @       �                                                                                                     
    @        wc���s    @                 !   Block
HIGH @���     ���             @    �   ���        @    ExpellingLeverActiveBlockHigh                 @       �                                                                                                     
    @        �q����    @                 !   Block
HIGH @���     ���             @    �   ���        @ )   ToExtractSpringInAssemblyStationBlockHigh                 @       �                                                                                                     
    @        ������    @                 !   Block
HIGH @���     ���             @    �   ���        @ #   PistonSelectorGoOnTheRightBlockHigh                 @       �                                                                                                     
    @        ������    @                 !   Block
HIGH @���     ���             @    �   ���        @ "   PistonSelectorGoOnTheLeftBlockHigh                 @       �                                                                                                     
    @        ���	��    @                 !   Block
HIGH @���     ���             @    �   ���        @ /   ToExtractCoverInAssemblyStationForwardBlockHigh                 @       �                                                                                                     
    @        ��1�!    @                 !   Block
HIGH @���     ���             @    �   ���        @ 1   BlockingCylinderForwardInAssemblyStationBlockHigh                 @       �                                                                                                     
    @        �8�X�H    @                 !   Block
HIGH @���     ���             @    �   ���        @ !   RobotGoToInitialPositionBlockHigh                 @       �                                                                                                     
    @        �_��o    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotGoToPistonHouseBlockHigh                 @       �                                                                                                     
    @        ������    @                 !   Block
HIGH @���     ���             @    �   ���        @    RobotGoToSpringHouseBlockHigh                 @       �                                                                                                     
    @        }q����    @                     Block
LOW @���     ���             @    �   ���        @ (   ToExtractSpringInAssemblyStationBlockLow                 @       �                                                                                                     
    @        }�����    @                     Block
LOW @���     ���             @    �   ���        @ "   PistonSelectorGoOnTheRightBlockLow                 @       �                                                                                                     
    @        }�����    @                     Block
LOW @���     ���             @    �   ���        @ !   PistonSelectorGoOnTheLeftBlockLow                 @       �                                                                                                     
    @        }��	��    @                     Block
LOW @���     ���             @    �   ���        @ .   ToExtractCoverInAssemblyStationForwardBlockLow                 @       �                                                                                                     
    @        }�1�!    @                     Block
LOW @���     ���             @    �   ���        @ 0   BlockingCylinderForwardInAssemblyStationBlockLow                 @       �                                                                                                     
    @        }8�X�H    @                     Block
LOW @���     ���             @        ���        @     RobotGoToInitialPositionBlockLow                 @       �                                                                                                     
    @        }_��m    @                     Block
LOW @���     ���             @       ���        @    RobotGoToPistonHouseBlockLow                 @       �                                                                                                     
    @        }�����    @                     Block
LOW @���     ���             @       ���        @    RobotGoToSpringHouseBlockLow                 @       �                                                                                                       
    @        � �(  ���     ���                                            RobotEngine @                         ���        @                       @                                                                                                         
    @        ~���    @                     Block
LOW @���     ���             @       ���        @    RobotEngineBlockLow                 @       �                                                                                                       
    @        ���(�  ���     ���                                            RobotGoToCoverHouse @                         ���        @                       @                                                                                                           
    @        ����)�  ���     ���                                         $   RobotTakeCurrentLoad
ToAssembly @                         ���        @                       @                                                                                                         
    @        ������    @                 !   Block
HIGH @���     ���             @       ���        @    RobotGoToCoverHouseBlockHigh                 @       �                                                                                                     
    @        ������    @                 !   Block
HIGH @���     ���             @       ���        @ '   RobotTakeCurrentLoadToAssemblyBlockHigh                 @       �                                                                                                     
    @        }�����    @                     Block
LOW @���     ���             @    	   ���        @    RobotGoToCoverHouseBlockLow                 @       �                                                                                                     
    @        ~�����    @                     Block
LOW @���     ���             @    
   ���        @ &   RobotTakeCurrentLoadToAssemblyBlockLow                 @       �         �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                  /   ,    j           FESTO_Interface ��d
    @    ��d     I                                                                                                     
    @        L��X�  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        � J u� #m   ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        � t����P��  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �lp�   ��     ���                                             @                      p   ���        @                       @                                                                                                           
    @        �������  ���     ���                                             @                      i   ���        @                       @                                                                                                           
    @        � |����8��  ���     ���                                             @                      S   ���        @                       @                                                                                                           
    @        � e� }� �  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �c�}��  ���     ���                                             @                      �   ���        @                       @                                                                                                         
    @        � R k� "l     @                 +   Fill All 
Warehouses @���     ���             @    ,   ���    	   Arial @        FillAllWarehouses             @       �                                                                                                     
    @        � �����7��    @                 #   Remove Pieces @���     ���             @    R   ���    	   Arial @        Remove             @       �                                                                                                       
    @        �|�4�����  ���     ���                                             @                      b   ���        @                       @                                                                                                         
    @        ���+�����    @                 !   Reset Fault @���     ���             @    U   ���    	   Arial @        FaultDetected             @       �                                                                                                       
    @        3~����f��  ��      ���                            	   NOT Fault        	   Fault @                      c   ���    	   Arial @                       @                                                                                                         
    @        oj����I   C:\DOCUMENTS AND SETTINGS\TOSHIBA\DESKTOP\3�ANNO\TESI\IMMAGINE2.BMP @                    Emergency @���     ���             @    g   ���        @    NOT EmergencyStopPuls                 @       �                                                                                                     
    @        &��������    @                 &   Simulate a Fault @���     ���             @    h   ���    	   Arial @                      @    Fault_Simulation  �                                                                                                       
    @        �����\��  ���     ���                                             @                      j   ���        @                       @                                                                                                         
    @        ������\��    @                 '   Show Fault's list @���     ���             @    k   ���    	   Arial @                      @    Fault_Detection  �                                                                                                     
    @        �sj�    @                 (   StopProgramRunning @���     ���             @    o   ���        @    StopProgramRunning                 @       �                                                                                                       
    @        e~������  ���     ���                                             @                      s   ���        @                       @                                                                                                         
    @        l��
�����    @                 $   Initial Values @���     ���             @    t   ���    	   Arial @                      @    Initial_Values  �                                                                                                       
    @        G}f�V�  ���     ���                                             @             MisuratorPosition        x   ���        @                       @                                                                                                           
    @        =}z�[�          ���     ���                     4   NOT ElementMeasureBlack OR NOT ElementMeasureCharged       ElementMeasureO   %s @         CylinderOfExtractionPosition            y   ���    	   Arial @                       @                                                                                                           
    @        =}z�[�  ���     ���                             5   NOT ElementMeasureSilver OR NOT ElementMeasureCharged       ElementMeasureO   %s @         CylinderOfExtractionPosition            z   ���    	   Arial @                       @                                                                                                           
    @        =}z�[�  �       ���                             2   NOT ElementMeasureRed OR NOT ElementMeasureCharged       ElementMeasureO   %s @         CylinderOfExtractionPosition            {   ���    	   Arial @                       @                                                                                                           
    @         ^ 1   ���     ���                                             @         CylinderPosition            |   ���        @                       @                                                                                                           
    @        S 	^ (X   ���     ���                                             @         CylinderPosition            }   ���        @                       @                                                                                                           
    @        �� ��  ���     �                                      VerificationBusy        @                      �   ���        @                       @                                                                                                           
    @        ��F�"  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        '*2,  �        �@                                     VirtualCylinderDownToMeasureLoad        @                      �   ���        @                       @                                                                                                         
    @        � �� 	��        ���     ���                                               RotaryPosition�   ���        @                                               @ 
    @           d                                                                                                          
    @         
     '  ' 
  
            ���                           @                         �   ���        @                                                                                                                              
    @         
  
   
  
    ���     ���                          @                         �   ���        @                             ��� �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                                                                                                           
    @        L�G�*  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @         1F� ;  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        ] 	� 2{           ���     ���                     ,   NOT ElementOneBlack OR NOT ElementOneCharged       ElementOneO   %s @         ElementPosition            �   ���    	   Arial @                       @                                                                                                           
    @        ] 	� 2{   ���     ���                             -   NOT ElementOneSilver OR NOT ElementOneCharged       ElementOneO   %s @         ElementPosition            �   ���    	   Arial @                       @                                                                                                           
    @        ] 	� 2{   �       ���                             *   NOT ElementOneRed OR NOT ElementOneCharged       ElementOneO   %s @         ElementPosition            �   ���    	   Arial @                       @                                                                                                           
    @        ] �� 
{ �  �       ���                             *   NOT ElementTwoRed OR NOT ElementTwoCharged       ElementTwoO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� 
{ �  ���     ���                             -   NOT ElementTwoSilver OR NOT ElementTwoCharged       ElementTwoO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� 
{ �          ���     ���                     ,   NOT ElementTwoBlack OR NOT ElementTwoCharged       ElementTwoO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] � B{ -  ���     ���                             1   NOT ElementSevenSilver OR NOT ElementSevenCharged       ElementSevenO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] � B{ -          ���     ���                     0   NOT ElementSevenBlack OR NOT ElementSevenCharged       ElementSevenO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] A� j{ U          ���     ���                     ,   NOT ElementSixBlack OR NOT ElementSixCharged       ElementSixO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �          ���     ���                     .   NOT ElementFourBlack OR NOT ElementFourCharged       ElementFourO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �          ���     ���                     0   NOT ElementThreeBlack OR NOT ElementThreeCharged       ElementThreeO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] i� �{ }          ���     ���                     .   NOT ElementFiveBlack OR NOT ElementFiveCharged       ElementFiveO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] A� j{ U  ���     ���                             -   NOT ElementSixSilver OR NOT ElementSixCharged       ElementSixO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] A� j{ U  �       ���                             *   NOT ElementSixRed OR NOT ElementSixCharged       ElementSixO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] i� �{ }  �       ���                             ,   NOT ElementFiveRed OR NOT ElementFiveCharged       ElementFiveO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �  �       ���                             ,   NOT ElementFourRed OR NOT ElementFourCharged       ElementFourO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �  �       ���                             .   NOT ElementThreeRed OR NOT ElementThreeCharged       ElementThreeO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] � B{ -  �       ���                             .   NOT ElementSevenRed OR NOT ElementSevenCharged       ElementSevenO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �  ���     ���                             1   NOT ElementThreeSilver OR NOT ElementThreeCharged       ElementThreeO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] i� �{ }  ���     ���                             /   NOT ElementFiveSilver OR NOT ElementFiveCharged       ElementFiveO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] �� �{ �  ���     ���                             /   NOT ElementFourSilver OR NOT ElementFourCharged       ElementFourO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @         �^ 20   ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @          2 '  �        �                                  +   CylinderExtractionLoadInRetroactivePosition        @                      �   ���        @                       @                                                                                                           
    @        I ^ 2S '  �        �                                  )   CylinderExtractionLoadInExtensivePosition        @                      �   ���        @                       @                                                                                                          
    @         � �� � � �� 	  ���     ���                          @                         �   ���        @                                                                                                                               
    @        � 	� 2�          ���     ���                     2   NOT ElementRotaryBlack OR NOT ElementRotaryCharged       ElementRotaryO   %s @                     RotaryPosition�   ���    	   Arial @                       @                                                                                                           
    @        � 	� 2�  �       ���                             0   NOT ElementRotaryRed OR NOT ElementRotaryCharged       ElementRotaryO   %s @                     RotaryPosition�   ���    	   Arial @                       @                                                                                                           
    @        � 	� 2�  ���     ���                             3   NOT ElementRotarySilver OR NOT ElementRotaryCharged       ElementRotaryO   %s @                     RotaryPosition�   ���    	   Arial @                       @                                                                                                           
    @        � ��� � � N    ��  �   ���                                             @                      �   ���        @                       @                                                                                                           
    @        r ���  �      ��  �   ���                                            Overturned @                      �   ���        @                       @                                                                                                           
    @        W;�FkE  �        �                                     RotaryMakerInPositionWarehouse        @                      �   ���        @                       @                                                                                                           
    @        �;�F�1  �        �                                  !   RotaryMakerInPositionVerification        @                      �   ���        @                       @                                                                                                           
    @         ��� ������ �� �  ���     ���                           @                RotaryPosition        �   ���        @                                                                                                                              
    @        ������          ���                                             @                      �   ���        @                       @                                                                                                           
    @        � ��U !    ��  �   ���                                         	   Short @                      �   ���        @                       @                                                                                                           
    @        H U9 N#    ��  �   ���                                             @                      �   ���        @                       @                                                                                                           
    @         ��� �� �� �� �  ���     ���                           @                RotaryPosition   NOT ExpulsionAirVacuumVis    �   ���        @                                                                                                                               
    @         ��� �� �� �� �  ���     ���                           @                RotaryPosition   NOT VacuumGeneratorOk    �   ���        @                                                                                                                               
    @        �*��  �        �@                                    CylinderUpToMeasureLoad        @                      �   ���        @                       @                                                                                                          
    @         �b�A�s�s  ���     ���                          @                         �   ���        @                                                                                                                              
    @         )�)E)A  ���     ���                          @                         �   ���        @                                                                                                                               
    @        )A�VVK  ��@     ���                                             @                      �   ���        @                       @                                                                                                           
    @        QU\~Qi  ���     ���                        MisuratorPosition                    @                      �   ���        @                       @                                                                                                           
    @        =	z2[          ���     ���                     >   NOT ElementVerificationBlack OR NOT ElementVerificationCharged       ElementVerificationO   %s @         CylinderOfExtractionPosition   -LiftPosition        �   ���    	   Arial @                       @                                                                                                           
    @        =	z2[  ���     ���                             ?   NOT ElementVerificationSilver OR NOT ElementVerificationCharged       ElementVerificationO   %s @         CylinderOfExtractionPosition   -LiftPosition        �   ���    	   Arial @                       @                                                                                                           
    @        =	z2[  �       ���                             <   NOT ElementVerificationRed OR NOT ElementVerificationCharged       ElementVerificationO   %s @         CylinderOfExtractionPosition   -LiftPosition        �   ���    	   Arial @                       @                                                                                                           
    @        )>23'  ���     ���                    CylinderOfExtractionPosition                        @             -LiftPosition        �   ���        @                       @                                                                                                          
    @         ��������  ���     ���                          @                         �   ���        @                                                                                                                               
    @        Q1fF[;  ���     ���                -LiftPosition                            @                      �   ���        @                       @                                                                                                           
    @        )1�FV;  ���     ���                                             @             -LiftPosition        �   ���        @                       @                                                                                                           
    @        )1\FB;  ��      ��                                             Load @                      �   ���        @                       @                                                                                                           
    @        )1\FB;  �       ���                                ReadyLoadForVerification        
   Unload @                      �   ���        @                       @                                                                                                          
    @         ������  ���     ���                          @                         �   ���        @                                                                                                                              
    @        QA�VjK  ���     ���                                            Measure @                      �   ���        @                       @                                                                                                          
    @        QA�VjK   �@     ���                             2   UncorrectComparison OR NOT CylinderUpToMeasureLoad           OK @                      �   ���        @                       @                                                                                                           
    @        [1�Fo;   �@                                                 
   Bright @             -LiftPosition        �   ���        @                       @                                                                                                           
    @        [1�Fo;                  ���                        ColourMeasurement        	   Black @             -LiftPosition        �   ���    	   Arial @	                       @                                                                                                          
    @        QA�VoK  �       ���                             6   NOT UncorrectComparison OR NOT CylinderUpToMeasureLoad        
   NOT OK @                      �   ���        @                       @                                                                                                           
    @        �  K9 #   ���     ���                                             @                      �   ���        @                       @                                                                                                         
    @        �  A0 "     @                    Silver S @���     ���             @    �   ���        @        Silvershort             @       �                                                                                                       
    @        , < s o O U   �       ���                                             @                      �   ���        @                       @                                                                                                           
    @        , l s � O �   ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        , 
 s = O #           ���                                             @                      �   ���        @                       @                                                                                                         
    @        6 x i � O �     @                    Silver @���     ���             @    �   ���        @        Silver             @       �                                                                                                     
    @        6  i 3 O #     @                    Black @���     ���             @    �   ���        @        Black             @       �                                                                                                     
    @        6 F i e O U     @                    Red @���     ���             @    �   ���        @        Red             @       �                                                                                                       
    @        r < � o � U   �       ���                                             @                      �   ���        @                       @                                                                                                           
    @        r l � � � �   ���     ���                                             @                      �   ���        @                       @                                                                                                         
    @        | F � e � U     @                    Red O @���     ���             @    �   ���        @        Redoverturned             @       �                                                                                                       
    @        r 
 � = � #           ���                                             @                      �   ���        @                       @                                                                                                         
    @        | x � � � �     @                    Silver O @���     ���             @    �   ���        @        Silveroverturned             @       �                                                                                                     
    @        |  � 3 � #     @                    Black O @���     ���             @    �   ���        @        Blackoverturned             @       �                                                                                                       
    @        '2,  ���      ��                                    CylinderDownToMeasureLoad        @                      �   ���        @                       @                                                                                                           
    @        � 	� 2�           ���     ���                     4   NOT ElementWaitingBlack OR NOT ElementWaitingCharged       ElementWaitingO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        � 	� 2�   ���     ���                             5   NOT ElementWaitingSilver OR NOT ElementWaitingCharged       ElementWaitingO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        � 	� 2�   �       ���                             2   NOT ElementWaitingRed OR NOT ElementWaitingCharged       ElementWaitingO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        )'42.,  �        ��                                 )   CylinderOfExtractionInRetroactivePosition        @             -LiftPosition        �   ���        @                       @                                                                                                          
    @        )ARV=K  � �     ���                                            High @                      �   ���        @                       @                                                                                                          
    @        )ARV=K          ���     ���                        MeasurementNotOk           Low @                      �   ���    	   Arial @	                       @                                                                                                           
    @        ] � � {   ���     ���                             1   NOT ElementEightSilver OR NOT ElementEightCharged       ElementEightO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] � � {   �       ���                             .   NOT ElementEightRed OR NOT ElementEightCharged       ElementEightO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        ] � � {           ���     ���                     0   NOT ElementEightBlack OR NOT ElementEightCharged       ElementEightO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        g 1� F{ ;   �  ��� �                                      EmptyWarehouse        @                      �   ���        @                       @                                                                                                           
    @        g 1� F{ ;   ��� ��� ���                                NOT EmptyWarehouse        	   Empty @                      �   ���        @                       @                                                                                                          
    @         { �] � ] 1� 1� 1  ���     ���                          @                         �   ���        @                                                                                                                               
    @        ��W�� ��  ���     ���                                             @                      �   ���        @                       @                                                                                                         
    @        ���O�� ��    @                 "   Button Panel @���     ���             @    �   ���    	   Arial @                      @    Switchboard  �                                                                                                       
    @        ��AN/   ���     ���                                         )   Tesi di Marco Pierantoni
a.a.2007-08 @                         ���    	   Arial @                       @                                                                                                          
    @        ��<  ��\     ���                                            @                         ���        @                       @                                                                                                           
    @        :�B�>�  ���     ���                                             @                         ���        @                       @                                                                                                         
    @        �*�S�>        ���     ���                                       ElementToDiscardPosition           ���        @         1   NOT PistonSilverToDiscard OR PistonSilverNumber<1                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                         ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �*�S�>        ���     ���                                       ElementToDiscardPosition        	   ���        @         /   NOT PistonBlackToDiscard OR PistonBlackNumber<1                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      
   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                          
    @        � 2=z�    ���     ���                                             @                         ���        @                       @                                                                                                         
    @        �:	%        ���     ���                                   PistonSelectorPosition               ���        @         -   NOT PistonBlackExtract OR PistonBlackNumber<1                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                         ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                          
    @        $5 "  d�Y     ���                                             @         PistonSelectorPosition               ���        @                       @                                                                                                         
    @        ��:�%        ���     ���                                   PistonSelectorPosition               ���        @         /   NOT PistonSilverExtract OR PistonSilverNumber<1                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                         ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                          
    @        ��5�"  d�Y     ���                                             @         PistonSelectorPosition               ���        @                       @                                                                                                           
    @        �P�c�Y   ���     ���                             &   NOT AvailableLoadInDrillingPositioning           Drilling Active @                         ���       Times New Roman @
                       @                                                                                                         
    @        �����        ���     ���                                                   ���        @         .   NOT PistonBlackAssembly OR PistonBlackNumber<0                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                         ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �����        ���     ���                                                   ���        @         0   NOT PistonSilverAssembly OR PistonSilverNumber<0                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                         ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                         ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                          
    @         � ������� � �
�
  ���     ���                           @                 $   NOT SpringAssembly OR SpringNumber<0       ���        @                                                                                                                              
    @         m:w&c0w0c:w:cDwD  ���     ���                           @    RobotGoHorizontalPosition   RobotGoVerticalPosition        )   NOT SpringInEndEffector OR SpringNumber<0       ���        @                                                                                                                             
    @        `)Ro=        ���     ���                                   RobotGoHorizontalPosition   RobotGoVerticalPosition           ���        @         3   NOT PistonBlackInEndEffector OR PistonBlackNumber<0                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                         ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                          ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        `)Rt=        ���     ���                                   RobotGoHorizontalPosition   RobotGoVerticalPosition        !   ���        @         5   NOT PistonSilverInEndEffector OR PistonSilverNumber<0                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      "   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      #   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                           
    @        �/�8�3  � �     ���            -ExtractCoverPosition                                @                      $   ���        @                       @                                                                                                           
    @        �1�Z�E  ���     ���     ���                     2   NOT ElementToDiscardSilver OR NOT ElementToDiscard       ElementToDiscardO   %s @             ElementToDiscardPosition        %   ���    	   Arial @                       @                                                                                                           
    @        �1�Z�E          ���     ���                     1   NOT ElementToDiscardBlack OR NOT ElementToDiscard       ElementToDiscardO   %s @             ElementToDiscardPosition        &   ���    	   Arial @                       @                                                                                                          
    @        ����  ���     ���                                             @                      '   ���        @                       @                                                                                                           
    @        ���F��  ���     ���                                             @                      (   ���        @                       @                                                                                                           
    @        �o�F��  ���     ���                                             @                      )   ���        @                       @                                                                                                           
    @        ������  � �     ���            BlockingActuator*-1                                @                      *   ���        @                       @                                                                                                           
    @        [� � o  fff     ���                                             @         RobotGoHorizontalPosition   RobotGoVerticalPosition        +   ���        @                       @                                                                                                           
    @        ���
  � �     ���                    CylinderPositionInAssemblyUnit                        @                      ,   ���        @                       @                                                                                                           
    @        ������  ���     ���                                             @                      -   ���        @                       @                                                                                                           
    @        q �M �3   ��h     ���            RobotGoHorizontalPosition                                @                      .   ���        @                       @                                                                                                           
    @         �� �= ��Z��0 �=   ��h     ���                           @                         /   ���        @                                                                                                                              
    @        ���
E�  ��h     ���                                            RV-M1   @                      0   ���    	   Arial @                       @                                                                                                           
    @        C�_FQ  ���     ���                                             @                      1   ���        @                       @                                                                                                           
    @        B�^�P�  ���     ���                                             @                      2   ���        @                       @                                                                                                           
    @        Qb8Y+  � �     ���            -ExtractSpringPosition                                @                      3   ���        @                       @                                                                                                           
    @        ��F�1  ���     ���                                             @                      4   ���        @                       @                                                                                                           
    @        ����  ���     ���                                NOT ElementAssemblySilver            @                     RotaryPosition*95   ���        @                       @                                                                                                           
    @        ����  �       ���                                NOT ElementAssemblyRed            @                     RotaryPosition*96   ���        @                       @                                                                                                           
    @        ����          ���                                NOT ElementAssemblyBlack            @                     RotaryPosition*97   ���        @                       @                                                                                                          
    @         G�P�<�P�<�P�<�P�  ���     ���                           @                    NOT Spring5    8   ���        @                                                                                                                              
    @         G�P�<�P�<�P�<�P�  ���     ���                           @                    NOT Spring4    9   ���        @                                                                                                                              
    @         G�P�<�P�<�P�<�P�  ���     ���                           @                    NOT Spring3    :   ���        @                                                                                                                              
    @         GUPD<NPN<XPX<bPb  ���     ���                           @                    NOT Spring7    ;   ���        @                                                                                                                              
    @         G�P�<�P�<�P�<�P�  ���     ���                           @                    NOT Spring2    <   ���        @                                                                                                                              
    @         FP�<P<P<P  ���     ���                           @                    NOT Spring1    =   ���        @                                                                                                                              
    @         :�:&::�  ���     ���                          @                         >   ���        @                                                                                                                              
    @         S�S$SS  ���     ���                          @                         ?   ���        @                                                                                                                             
    @        ��
��        ���     ���                                                @   ���        @            NOT PistonBlack1                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      A   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      B   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �q
��{        ���     ���                                                C   ���        @            NOT PistonBlack4                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      D   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      E   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        ��
���        ���     ���                                                F   ���        @            NOT PistonBlack3                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      G   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      H   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        ��
���        ���     ���                                                I   ���        @            NOT PistonBlack2                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      J   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      K   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �I
r�]        ���     ���                                                L   ���        @            NOT PistonBlack5                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      M   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      N   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �� 
� ��         ���     ���                                                O   ���        @            NOT PistonBlack8                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      P   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      Q   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �� 
"�        ���     ���                                                R   ���        @            NOT PistonBlack7                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      S   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      T   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �!
J�5        ���     ���                                                U   ���        @            NOT PistonBlack6                                  @ 
    @            d                                                                                                          
    @            ) 
            ���                                             @                      V   ���        @                       @                                                                                                           
    @        
    ) 
            ���                                             @                      W   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �� �� ��         ���     ���                                                X   ���        @            NOT PistonSilver8                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      Y   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      Z   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �� �"�        ���     ���                                                [   ���        @            NOT PistonSilver7                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      \   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      ]   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �!�J�5        ���     ���                                                ^   ���        @            NOT PistonSilver6                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      _   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      `   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �I�r�]        ���     ���                                                a   ���        @            NOT PistonSilver5                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      b   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      c   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �q����        ���     ���                                                d   ���        @            NOT PistonSilver4                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      e   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      f   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        ������        ���     ���                                                g   ���        @            NOT PistonSilver3                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      h   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      i   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        ������        ���     ���                                                j   ���        @            NOT PistonSilver2                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      k   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      l   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �����        ���     ���                                                m   ���        @            NOT PistonSilver1                                  @ 
    @            d                                                                                                          
    @        
    )     ���     ���                                             @                      n   ���        @                       @                                                                                                           
    @            )  #   ���     ���                                             @                      o   ���        @                       @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                          
    @         �L��� ��   ���     ���                          @                         p   ���        @                                                                                                                              
    @         X�   ���     ���                          @                         q   ���        @                                                                                                                              
    @         �W����   ���     ���                          @                         r   ���        @                                                                                                                              
    @         �R����   ���     ���                          @                         s   ���        @                                                                                                                              
    @         FxPd<nPn<xPx<�P�  ���     ���                           @                    NOT Spring6    t   ���        @                                                                                                                               
    @        ������    �     ���                             
   NOT Cover7            @                      u   ���        @                       @                                                                                                           
    @        ����    �     ���                             
   NOT Cover6            @                      v   ���        @                       @                                                                                                           
    @        ���    �     ���                             
   NOT Cover5            @                      w   ���        @                       @                                                                                                           
    @        ���    �     ���                             
   NOT Cover4            @                      x   ���        @                       @                                                                                                           
    @        ��%�    �     ���                             
   NOT Cover3            @                      y   ���        @                       @                                                                                                           
    @        �$�/�)    �     ���                             
   NOT Cover2            @                      z   ���        @                       @                                                                                                           
    @        �.�9�3    �     ���                             
   NOT Cover1            @                      {   ���        @                       @                                                                                                           
    @        ������    �     ���                             
   NOT Cover8            @                      |   ���        @                       @                                                                                                           
    @        H.9c3    �     ���                             !   NOT CoverExtract OR CoverNumber<0            @                      }   ���        @                       @                                                                                                           
    @        ������    �     ���                             "   NOT CoverAssembly OR CoverNumber<0            @                      ~   ���        @                       @                                                                                                           
    @        i��F�  ���     ���                                            @                         ���        @                       @                                                                                                           
    @        LiT�P�  @��     ���                                             @             InspectPosition        �   ���        @                       @                                                                                                           
    @         ,$�8�8E$E$1  ���     ���                           @                         �   ���        @                                                                                                                               
    @        cxFm+  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        S��o�  �       ���     ���                     >   NOT ElementStation1RobotRed OR NOT ElementStation1RobotCharged       ElementStation1RobotO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        S��o�          ���     ���                     @   NOT ElementStation1RobotBlack OR NOT ElementStation1RobotCharged       ElementStation1RobotO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        S��o�  ���     ���     ���                     A   NOT ElementStation1RobotSilver OR NOT ElementStation1RobotCharged       ElementStation1RobotO   %s @                      �   ���    	   Arial @                       @                                                                                                           
    @        K�m  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        eum  �        �                                     AvailableLoadForRobot        @                      �   ���        @                       @                                                                                                           
    @        �� %
   ���     ���                                NOT ExpellingLeverActive           Expulsion
   Lever @                      �   ���    	   Arial @
                       @                                                                                                           
    @        P�cPY   ���     ���                             %   NOT AvailableLoadInControlPositioning           Control Active @                      �   ���       Times New Roman @
                       @                                                                                                           
    @        � %? {/    ���     ���                                         '   VIEW FROM ABOVE OF THE ROTARY TABLE @                      �   ���    	   Arial @                       @                                                                                                          
    @        �v^�$�  ��\     ���                                             @                      �   ���        @                       @                                                                                                          
    @        � �^ �5   ��\     ���                                             @                      �   ���        @                       @                                                                                                          
    @        ��Q�%�  ��\     ���                                             @                      �   ���        @                       @                                                                                                           
    @        ����  ~~~     ���                                             @                      �   ���        @                       @                                                                                                          
    @         ���+����  ���     ���                          @                         �   ���        @                                                                                                                              
    @         ���+����  ���     ���                          @                         �   ���        @                                                                                                                              
    @         �,�,�,�,  ���     ���                          @                         �   ���        @                                                                                                                               
    @        P�#o  fff     ���            EndEffectorPosition       -EndEffectorPosition                        @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���        @                       @                                                                                                           
    @        ��9�(  fff     ���                                             @      /   (RobotGoHorizontalPosition)-EndEffectorPosition   RobotGoVerticalPosition        �   ���        @                       @                                                                                                           
    @        LR9O(  fff     ���                                             @      /   (RobotGoHorizontalPosition)+EndEffectorPosition   RobotGoVerticalPosition        �   ���        @                       @                                                                                                           
    @        T3�\pG  �       ���     ���                     7   NOT ElementInEndEffectorRed OR NOT ElementInEndEffector       ElementInEndEffectorO   %s @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���    	   Arial @                       @                                                                                                           
    @        T3�\pG          ���     ���                     9   NOT ElementInEndEffectorBlack OR NOT ElementInEndEffector       ElementInEndEffectorO   %s @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���    	   Arial @                       @                                                                                                           
    @        T3�\pG  ���     ���     ���                     :   NOT ElementInEndEffectorSilver OR NOT ElementInEndEffector       ElementInEndEffectorO   %s @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���    	   Arial @                       @                                                                                                           
    @         8�&��<�T�T�  �       ���     ���                   @    MovementElementSleigh*2   MovementElementSleigh        0   NOT ElementSleighRed OR NOT ElementSleighCharged    �   ���    	   Arial @            ElementAirO   %s                                                                                                              
    @         8�&��<�T�T�  ���     ���     ���                   @    MovementElementSleigh*2   MovementElementSleigh        3   NOT ElementSleighSilver OR NOT ElementSleighCharged    �   ���    	   Arial @            ElementAirO   %s                                                                                                              
    @         8�&��<�T�T�          ���     ���                   @    MovementElementSleigh*2   MovementElementSleigh        2   NOT ElementSleighBlack OR NOT ElementSleighCharged    �   ���    	   Arial @            ElementSleighO   %s                                                                                                             
    @        � �U �5   ��\     ���                                             @                      �   ���        @                       @                                                                                                          
    @        ��FM,  ��h     ���     �                                      MITSUBISHI @                      �   ���    	   Arial @	                       @                                                                                                           
    @         5���KKK�  ���     ���                           @                         �   ���        @                                                                                                                               
    @        [8 �� o�   ��h     ���                        RobotGoVerticalPosition                    @         RobotGoHorizontalPosition            �   ���        @                       @                                                                                                          
    @        M �Y p7   ��\     ���                                             @         RobotGoHorizontalPosition            �   ���        @                       @                                                                                                          
    @        U �Q p6   ��\     ���                                             @         RobotGoHorizontalPosition            �   ���        @                       @                                                                                                          
    @         H6Q%=/Q/=9Q9=CQC  ���     ���                           @                    NOT Spring8    �   ���        @                                                                                                                              
    @         .($($.(.8(8  ���     ���                           @                 #   NOT SpringExtract OR SpringNumber<0    �   ���        @                                                                                                                              
    @         F.P<$P$<.P.<8P8  ���     ���                           @    -ExtractSpringPosition               NOT VisualSpring    �   ���        @                                                                                                                              
    @        !=�G�B  ���     ���                                            @                      �   ���        @                       @                                                                                                          
    @        �=�I�C  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        =FI/C  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        �=�I�C  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        W<�HnB  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        ���  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        =9�J�A  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        :�C�>  �        �                                  2   ToExtractCoverInAssemblyStationInExtensivePosition        @                      �   ���        @                       @                                                                                                           
    @        �:�C�>  �        �                                  4   ToExtractCoverInAssemblyStationInRetroactivePosition        @                      �   ���        @                       @                                                                                                           
    @        �"�M�7  ~~~     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �+�H�9  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        �.�8�3  �        �                                      EmptyCoverHouseInAssemblyStation        @                      �   ���        @                       @                                                                                                          
    @        [� �o�   ��\     ���                                             @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���        @                       @                                                                                                           
    @        ��#�  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @         �	�	�	�	  ���     ���                          @                         �   ���        @                                                                                                                              
    @         V\TT  ���     ���                          @                         �   ���        @                                                                                                                               
    @        Zm;c#  ~~~     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �l�t�p  �        �                                     DrillingUnitUp        @                      �   ���        @                       @                                                                                                           
    @        ������  �        �                                     DrillingUnitDown        @                      �   ���        @                       @                                                                                                          
    @        ��9�$  d�Y     ���                                             @         PistonSelectorPosition            �   ���        @                       @                                                                                                           
    @        �+$:�2  d�Y     ���                                             @         PistonSelectorPosition            �   ���        @                       @                                                                                                           
    @        �
�:�"  [[[     ���                                             @                      �   ���        @                       @                                                                                                           
    @        r97Q�E  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �9B�=  �        �                                     PistonSelectorIsOnTheRight        @                      �   ���        @                       @                                                                                                           
    @        �9�B�=  �        �                                     PistonSelectorIsOnTheLeft        @                      �   ���        @                       @                                                                                                           
    @        	:mF;@  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        K:ZCR>  �        �                                  5   ToExtractSpringInAssemblyStationInRetroactivePosition        @                      �   ���        @                       @                                                                                                           
    @        ":1C)>  �        �                                  3   ToExtractSpringInAssemblyStationInExtensivePosition        @                      �   ���        @                       @                                                                                                           
    @        T.�9p3    �     ���                             '   NOT CoverInEndEffector OR CoverNumber<0            @         RobotGoHorizontalPosition   RobotGoVerticalPosition        �   ���        @                       @                                                                                                           
    @        �� �    ���     ���                             )   NOT AlignementRotaryTableWithPositionings            Table  
Aligned @                      �   ���    	   Arial @
                       @                                                                                                           
    @        �� � ��   ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �� *� �   ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        �� � ��    ��     ���                                NOT AirCushion            @                      �   ���        @                       @                                                                                                          
    @        �� �� ��           ���                             ,   NOT ElementAirBlack OR NOT ElementAirCharged            @         MovementElementAir            �   ���        @                       @                                                                                                          
    @        �� �� ��   ���     ���                             -   NOT ElementAirSilver OR NOT ElementAirCharged            @         MovementElementAir            �   ���        @                       @                                                                                                          
    @        �� �� ��   �       ���                             *   NOT ElementAirRed OR NOT ElementAirCharged            @         MovementElementAir            �   ���        @                       @                                                                                                          
    @        �� � ��           ���                             2   NOT ElementSleighBlack OR NOT ElementSleighCharged            @         MovementElementSleigh            �   ���        @                       @                                                                                                          
    @        �� � ��   ���     ���                             3   NOT ElementSleighSilver OR NOT ElementSleighCharged            @         MovementElementSleigh            �   ���        @                       @                                                                                                          
    @        �� � ��   �       ���                             0   NOT ElementSleighRed OR NOT ElementSleighCharged            @         MovementElementSleigh            �   ���        @                       @                                                                                                           
    @        I�W�P�  @��     ���                                             @             InspectPosition        �   ���        @                       @                                                                                                           
    @        Bg^oPk  @��     ���                                             @             InspectPosition        �   ���        @                       @                                                                                                           
    @        �1�Z�E  �       ���     ���                     /   NOT ElementToDiscardRed OR NOT ElementToDiscard       ElementToDiscardO   %s @             ElementToDiscardPosition        �   ���    	   Arial @                       @                                                                                                           
    @        �.�9�3    �     ���                             #   NOT CoverToDiscard OR CoverNumber<0            @             ElementToDiscardPosition        �   ���        @                       @                                                                                                           
    @        �h ���   �   ���  �@     ���                        NOT CanDiscard	   CanColour   CanText   %s @                      �   ���    	   Arial @
                       @                                                                                                           
    @         ���e�����e   ���     ���                           @                    NOT CanDiscard    �   ���        @                                                                                                                               
    @          �ee���   ���     ���                           @                    NOT CanDiscard    �   ���        @                                                                                                                               
    @        ������  ���     ���                                NOT CanDiscard            @                      �   ���        @                       @                                                                                                           
    @        :�g�P�  ���     ���                                             @                      �   ���        @                       @                                                                                                           
    @        9yB�=}  �        �                                  '   InControlLoadInWrongPositionToBeDrilled        @                      �   ���        @                       @                                                                                                           
    @        �����  �        �                                     LightRobotInMovement        @                      �   ���        @                       @                                                                                                         
    @        �|����        ���     ���                                       DrillingUnitPosition        �   ���        @            NOT DrillingToolColor                                  @ 
    @            d                                                                                                          
    @                   2  2   2   ��@     ���                           @                         �   ���        @                                                                                                                               
    @          2   2  8  3  -   ��@     ���                           @                         �   ���        @                                                                                                                              
    @          '   +  &  &   ���     ���                           @                         �   ���        @                                                                                                                              
    @              $       ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                                                                                                                              
    @               
  
   ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                             ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                         
    @        �|����        ���     ���                                       DrillingUnitPosition        �   ���        @            DrillingToolColor                                  @ 
    @           d                                                                                                          
    @                   2  2   2   � �     ���                           @                         �   ���        @                                                                                                                               
    @          2   2  8  3  -   � �     ���                           @                         �   ���        @                                                                                                                              
    @          '   +  &  &   ���     ���                           @                         �   ���        @                                                                                                                              
    @              $       ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                                                                                                                              
    @               
  
   ���     ���                           @                         �   ���        @                                                                                                                              
    @                    ���     ���                           @                         �   ���        @                             ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���                                                                                                           
    @        �s����  ���     ���                                             @             DrillingUnitPosition/2        �   ���        @                       @                                                                            Color6                              
    @        �����  �       ���     ���                        Color6=ColorCircle[2]       ElementSixTableO   %s @         LinearTablePosition6            �   ���    	   Arial @                       @                                                                            Color5                              
    @        �����  �       ���     ���                        Color5=ColorCircle[2]       ElementFiveTableO   %s @         LinearTablePosition5            �   ���    	   Arial @                       @                                                                            Color4                              
    @        �����  �       ���     ���                        Color4=ColorCircle[2]       ElementFourTableO   %s @         LinearTablePosition4            �   ���    	   Arial @                       @                                                                            Color3                              
    @        �����  �       ���     ���                        Color3=ColorCircle[2]       ElementThreeTableO   %s @         LinearTablePosition3            �   ���    	   Arial @                       @                                                                            Color2                              
    @        �����  �       ���     ���                        Color2=ColorCircle[2]       ElementTwoTableO   %s @         LinearTablePosition2            �   ���    	   Arial @                       @                                                                            Color1                              
    @        �����  �       ���     ���                        Color1=ColorCircle[2]       ElementOneTableO   %s @         LinearTablePosition1            �   ���    	   Arial @                       @                                                                                                          
    @        �  �   �        �                                  %   AlignementRotaryTableWithPositionings        @                      �   ���        @                       @                                                                                                          
    @        M �*{�   ���     ���                                            @                      �   ���        @                       @                                                                                                          
    @        � � {�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                                                          
    @        D` Ok {�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                                                          
    @        �� �� {�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                                                          
    @        �` �k {�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                                                          
    @        ��{�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                                                          
    @        DO{�   ���     ���                                             @                     RotaryTableVisual�   ���        @                       @                                                                           Color2                              
    @        j� �{�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                           Color3                              
    @        �� �� {�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                           Color4                              
    @        �� �� {�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                           Color5                              
    @        k] �} {�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                           Color6                              
    @        &� H� {�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                           Color1                              
    @        %� G� {�   ���     ���                                            @                     RotaryTableVisual�   ���        @                       @                                                                                                           
    @         �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��   � �     ���                           @                LeverPosition        �   ���        @                                                                                                                              
    @        �� �� ��    @�     ���                                             @                          ���        @                       @                                                                                                          
    @        t� �� {�   ***      �                                              @                         ���        @                       @                                                                                                           
    @        ���y�  ���     ���                                            @                         ���        @                       @                                                                                                           
    @        �����  �        �                                     AvailableLoadForWorkingStation        @                         ���        @                       @                                                                                                           
    @        G�\�Q�  �        �                                  !   AvailableLoadInControlPositioning        @                         ���        @                       @                                                                                                           
    @        ������  �        �                                  "   AvailableLoadInDrillingPositioning        @                         ���        @                       @                                                                                                           
    @        �
7�  ���     ���                                             @                         ���        @                       @                                                                                                          
    @        � �9�  ���     ��                                     RotaryTableMotor        @                         ���        @                       @                                                                                                           
    @        �$�   ���     ���     ��                         NOT RotaryTableMotor           RotaryTableMotor @                      	   ���    	   Arial @                       @                                                                                                           
    @        F'  ���     ���                                             @                      
   ���        @                       @                                                                                                           
    @        �  �        �                                  %   AlignementRotaryTableWithPositionings        @                         ���        @                       @                                                                                                           
    @         D�i�V  ���     ���                                             @                      �   ���        @                       @                                                                                                          
    @        � 2� �Q   ���     ���                                             @                         ���        @                       @                                                                                                          
    @        �  'O 2   �       ���     ���                                   STOP @                         ���    	   Arial @	        Stop            @                                                                                                          
    @        �Z )� s   ���     ��                                 
   LightReset    	   RESET @                         ���    	   Arial @	        Reset             @                                                                                                          
    @        �_ �� �x   ���     ��                                 
   LightStart    	   START @                         ���    	   Arial @	        Start             @                                                                                                         
    @        � �Y �9 I   C:\DOCUMENTS AND SETTINGS\TOSHIBA\DESKTOP\3�ANNO\TESI\IMMAGINE3.BMP @                    FREEZE @���     ���             @       ���    	   Arial @    NOT FreezeStopPuls                 @       �                                                                                                       
    @         ���p~�������  �       ���     ���                   @    MovementElementAir*2   MovementElementAir        *   NOT ElementAirRed OR NOT ElementAirCharged       ���    	   Arial @            ElementAirO   %s                                                                                                              
    @         ���p~�������  ���     ���     ���                   @    MovementElementAir*2   MovementElementAir        -   NOT ElementAirSilver OR NOT ElementAirCharged       ���    	   Arial @            ElementAirO   %s                                                                                                              
    @         ���p~�������          ���     ���                   @    MovementElementAir*2   MovementElementAir        ,   NOT ElementAirBlack OR NOT ElementAirCharged       ���    	   Arial @            ElementAirO   %s                                                                                                              
    @         ������������  ���     ���                           @                         �   ���        @                                                                                                                               
    @         ������������  ���     ���                           @                    NOT AirCushion    �   ���        @                                                                                                                              
    @        |� O� ��   ���      �@                                    ResetSignalsEnable       RESET_BUTTON @                         ���        @        Reset_Phisical             @         ��� �   ��   �   ��   � � � ��� ��� �   ��   �   ��   � � � ���      ����                 V   , ! ��        	   HMI_Panel ��d
    @�����R�d�     G                                                                                                       
    @        � � � 7� �   ���     ��                                              @                           ���        @	                       @                                                                                                           
    @        � � � � � �    �� @ @ @                                       "   Memory_Data[Distribution_index].ID   %d @                      
    ���    	   Arial @                       @                                                                                                           
    @        < � � � i �   ���     ���                                            ID @                          ���    	   Arial @                       @                                                                                                           
    @        < � � � i �   ���     ���                                         
   Colour @                          ���    	   Arial @                       @                                                                                                           
    @        < � � � i �   ���     ���                                         
   Height @                          ���    	   Arial @                       @                                                                                                           
    @        � � � � � �   ���     ���                                            Distribution @                          ���    	   Arial @                       @                                                                                                           
    @        � � _7,�   ���     ��                                              @                          ���        @	                       @                                                                                                           
    @        � U� ,�           �                                  Memory_Data[Testing_index].ID=0!   Memory_Data[Testing_index].Height        @                          ���        @	                       @                                                                                                           
    @        � U� ,�       @ @ �                                  Memory_Data[Testing_index].ID=0!   Memory_Data[Testing_index].Colour        @                          ���        @	                       @                                                                                                           
    @        � U� ,�    �� @ @ @                                          Memory_Data[Testing_index].ID   %d @                          ���    	   Arial @                       @                                                                                                           
    @        � � _� ,�   ���     ���                                            Testing @                          ���    	   Arial @                       @                                                                                                           
    @        ^� �7��   ���     ��                                              @                          ���        @	                       @                                                                                                           
    @        h� �� ��           �                                  Memory_Data[Rotary_index].ID=0    Memory_Data[Rotary_index].Height        @                          ���        @	                       @                                                                                                           
    @        h� �� ��       @ @ �                                  Memory_Data[Rotary_index].ID=0    Memory_Data[Rotary_index].Colour        @                          ���        @	                       @                                                                                                           
    @        h� �� ��    �� @ @ @                                          Memory_Data[Rotary_index].ID   %d @                          ���    	   Arial @                       @                                                                                                           
    @        ^� �� ��   ���     ���                                         
   Rotary @                          ���    	   Arial @                       @                                                                                                           
    @        �� '7��   ���     ��                                              @                           ���        @	                       @                                                                                                           
    @        �� � ��           �                               "   Memory_Data[Inspection_index].ID=0$   Memory_Data[Inspection_index].Height        @                      !    ���        @	                       @                                                                                                           
    @        �� � ��       @ @ �                               "   Memory_Data[Inspection_index].ID=0$   Memory_Data[Inspection_index].Colour        @                      "    ���        @	                       @                                                                                                           
    @        �� � ��   ��� @ @  ��                                         Memory_Data[Inspection_index].ID   %d @                      #    ���    	   Arial @                       @                                                                                                           
    @        �� '� ��   ���     ���                                            Inspection @                      '    ���    	   Arial @                       @                                                                                                           
    @        &� �7X�   ���     ��                                              @                      (    ���        @	                       @                                                                                                           
    @        0� �� X�           �                                   Memory_Data[Drilling_index].ID=0"   Memory_Data[Drilling_index].Height        @                      )    ���        @	                       @                                                                                                           
    @        0� �� X�       @ @ �                                   Memory_Data[Drilling_index].ID=0"   Memory_Data[Drilling_index].Colour        @                      *    ���        @	                       @                                                                                                           
    @        0� �� X�    �� @ @ @                                          Memory_Data[Drilling_index].ID   %d @                      +    ���    	   Arial @                       @                                                                                                           
    @        &� �� X�   ���     ���                                            Drilling @                      /    ���    	   Arial @                       @                                                                                                           
    @        �� �7��   ���     ��                                              @                      0    ���        @	                       @                                                                                                           
    @        �� �� ��           �                               !   Memory_Data[Expelling_index].ID=0#   Memory_Data[Expelling_index].Height        @                      1    ���        @	                       @                                                                                                           
    @        �� �� ��       @ @ �                               !   Memory_Data[Expelling_index].ID=0#   Memory_Data[Expelling_index].Colour        @                      2    ���        @	                       @                                                                                                           
    @        �� �� ��    �� @ @ @                                          Memory_Data[Expelling_index].ID   %d @                      3    ���    	   Arial @                       @                                                                                                           
    @        �� �� ��   ���     ���                                            Expelling @                      7    ���    	   Arial @                       @                                                                                                           
    @        �� S7 �   ���     ��                                              @                      H    ���        @	                       @                                                                                                           
    @        �� I�  �           �                               $   Memory_Data[PickandPlace_index].ID=0&   Memory_Data[PickandPlace_index].Height        @                      I    ���        @	                       @                                                                                                           
    @        �� I�  �       @ @ �                               $   Memory_Data[PickandPlace_index].ID=0&   Memory_Data[PickandPlace_index].Colour        @                      J    ���        @	                       @                                                                                                           
    @        �� I�  �    �� @ @ @                                       "   Memory_Data[PickandPlace_index].ID   %d @                      K    ���        @	                       @                                                                                                           
    @        �� S�  �   ���     ���                                            PickandPlace @                      O    ���    	   Arial @                       @                                                                                                           
    @         �3 Y#   ���     ��                                 Main_PRG.state<>0           Ready to initialize @                      Y    ���    	   Arial @                      @                                                                                                           
    @        � 3 �#   ���     ��                                 Main_PRG.state<>3           Running @                      Z    ���    	   Arial @                       @                                                                                                           
    @        � 	3 �#   ���     ��                                 Main_PRG.state<>1           Initializing @                      ]    ���    	   Arial @                       @                                                                                                           
    @        R� �7��   ���     ��                                              @                      a   ���        @	                       @                                                                                                           
    @        R� �� ��   ���     ���                                            Assembly @                      b   ���    	   Arial @                       @                                                                                                           
    @        \� �� ��           �                                  Memory_Data[Supply_index].ID=0    Memory_Data[Supply_index].Height        @                      ^   ���        @	                       @                                                                                                           
    @        \� �� ��       @ @ �                                  Memory_Data[Supply_index].ID=0    Memory_Data[Supply_index].Colour        @                      _   ���        @	                       @                                                                                                           
    @        \� �� ��    �� @ @ @                                          Memory_Data[Supply_index].ID   %d @                      `   ���    	   Arial @                       @                                                                                                           
    @        � a3    ���     ��                                 Main_PRG.state<>5           On Phase Stopping @                      l   ���    	   Arial @                       @                                                                                                           
    @        <  3 � #   ���     ���                                            Main Program States @                      �   ���       Arial Narrow @                       @                                                                                                           
    @        �Z Oy �n   ���     ���                                         
   Buffer @                      �   ���       Arial Narrow @                       @                                                                                                           
    @        < � � i   ���     ���                                            Orientation @                      �   ���    	   Arial @                       @                                                                                                           
    @        < � -i "  ���     ���                                            Discard @                      �   ���    	   Arial @                       @                                                                                                           
    @        U-,"   �      �                                  Memory_Data[Testing_index].ID=0"   Memory_Data[Testing_index].Discard        @                      �   ���        @	                       @                                                                                                           
    @        �� �  ��       �                              "   Memory_Data[Inspection_index].ID=0)   Memory_Data[Inspection_index].Orientation        @                      �   ���        @	                       @                                                                                                           
    @        �-�"   �      �                               "   Memory_Data[Inspection_index].ID=0%   Memory_Data[Inspection_index].Discard        @                      �   ���        @	                       @                                                                                                           
    @        0� �X  ��       �                                  Memory_Data[Drilling_index].ID=0'   Memory_Data[Drilling_index].Orientation        @                      �   ���        @	                       @                                                                                                           
    @        0�-X"   �      �                                   Memory_Data[Drilling_index].ID=0#   Memory_Data[Drilling_index].Discard        @                      �   ���        @	                       @                                                                                                           
    @        �� ��  ��       �                              !   Memory_Data[Expelling_index].ID=0(   Memory_Data[Expelling_index].Orientation        @                      �   ���        @	                       @                                                                                                           
    @        ��-�"   �      �                               !   Memory_Data[Expelling_index].ID=0$   Memory_Data[Expelling_index].Discard        @                      �   ���        @	                       @                                                                                                           
    @        �� I   ��       �                              $   Memory_Data[PickandPlace_index].ID=0+   Memory_Data[PickandPlace_index].Orientation        @                      �   ���        @	                       @                                                                                                           
    @        �I- "   �      �                               $   Memory_Data[PickandPlace_index].ID=0'   Memory_Data[PickandPlace_index].Discard        @                      �   ���        @	                       @                                                                                                           
    @        \� ��  ��       �                                 Memory_Data[Supply_index].ID=0%   Memory_Data[Supply_index].Orientation        @                      �   ���        @	                       @                                                                                                           
    @         �3 f#   ���     ��                                 Main_PRG.state<>4           Immediate Stopping @                      �   ���    	   Arial @                       @                                                                                                          
    @        < �y �Z �  ���      �                                     Main_PRG.state=0       Init @                      �   ���    	   Arial @        Init_HMI             @                                                                                                          
    @        � �� �� �  ���      �                                     LightStartLogical    	   Start @                      �   ���    	   Arial @     	   Start_HMI             @                                                                                                          
    @        < �� � �  ���     �                                      OnPhaseStop_Logical       On Phase Stop @                      �   ���    	   Arial @    OnPhaseStop_HMI   OnPhaseStop_HMI             @                                                                                                          
    @        < &� Y� ?  ���     �                                      ImmediateStop_Logical       Immediate Stop @                      �   ���    	   Arial @    ImmediateStop_HMI   ImmediateStop_HMI             @                                                                                                          
    @        � �A1  ���      �                                     ResetSignalsEnable       Reset Fault @                      �   ���    	   Arial @     	   Reset_HMI             @                                                                                                           
    @        d h�� |  ���     ���                                            Buttons Panel @                      �   ���       Arial Narrow @                       @                                                                                                           
    @        &h��v|  ���     ���                                            Lights Panel @                      �   ���       Arial Narrow @                       @                                                                                                          
    @        ���{�  ���     ��                                     LightEmptyWarehouseLogical       Empty Warehouse @                      �   ���    	   Arial @                       @                                                                                                          
    @        ��{�  ���     ��                                     LightEmptyCoverhouseLogical       Empty Cover House @                      �   ���    	   Arial @                       @                                                                                                           
    @         �3 S#   ���     ��                                 Main_PRG.state<>2           Ready to Run @                      �   ���    	   Arial @                       @                                                                                                           
    @        t �3 �#   ���     ��                                 Main_PRG.state<>6           Stopping @                      �   ���    	   Arial @                       @             �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                  4   , � � �>           Initial_Values ��d
    @    ��d   d                                                                                                          
    @          1� � m   ���  �� ���                                            @                          ���        @	                       @                                                                                                     0   20   Valori @        @ E 	x � ^   ���     ���                                        RotaryPositionInitialVis'   Valore iniziale braccio rotante: %d @                          ���        @	                      @                                                                                                     0   20   Valori @        @ � 	� � �   ���     ���                                        LiftPositionInitialVis!   Valore iniziale ascensore: %d @                          ���        @	                      @                                                                                                           
    @        3  ; � +    ���     ���                                         ,   Valori Iniziali Distribuzione e Verifica @                          ���    	   Arial @	                       @                                                                                                           
    @        H c���   ���  �� ���                                            @                          ���        @	                       @                                                                                                     0   30   Valori @        qE :x �^   ���     ���                                        RotaryVisIn&   Valore iniziale Tavola rotante: %d @                      	    ���        @	                      @                                                                                                           
    @        L [< �,    ���     ���                                         .   Valori Iniziali Lavorazione e Assemblaggio @                          ���    	   Arial @	                       @                                                                                                     0   20   Valori @        q� 9� ��   ���     ���                                        DrillingUnitPosition3   Posizione iniziale dell'unit� 
di foratura: %d @                          ���        @	                      @                                                                                                     -20   20   Valori @        s� 9� ��   ���     ���                                        PistonSelectorPosition?   Posizione iniziale del cilindro 
di estrazione pistoni: %d @                          ���        @	                      @                                                                                                     -100   100   Valori @        s94�  ���     ���                                        RobotGoHorizontalPosition'   Posizione Orizzontale del Robot: %d @                          ���        @	                      @                                                                                                     -100   100   Valori @        s>9q�W  ���     ���                                        RobotGoVerticalPosition%   Posizione Verticale del Robot: %d @                          ���        @	                      @                                                                                                         
    @        J ?� '    @                 #   Back to Plant @���     ���             @        ���    	   Arial @                      @    FESTO_Interface  �         �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                  ?   , Y����           Switchboard ��d
    @    ��d�   (   ;                                                                                                       
    @        � z���   ���  �� ���                                            @                      �    ���        @	                       @                                                                                                           
    @          ���   ���  �� ���                                            @                      �    ���        @	                       @                                                                                                           
    @        ( � � _�   ���   � ���                                             @                      �    ���        @	                       @                                                                                                           
    @        �Z �� I�   ���   � ���                                             @                      w    ���        @	                       @                                                                                                           
    @        �Z g� �   ���   � ���                                             @                      x    ���        @	                       @                                                                                                           
    @        ( ^� �� �  ���   � ���                                             @                      q    ���        @	                       @                                                                                                           
    @        � � �_E  ���   � ���                                             @                      s    ���        @	                       @                                                                                                           
    @        ( Z � � � �   ���   � ���                                             @                      t    ���        @	                       @                                                                                                           
    @        � Z �� E�   ���   � ���                                             @                      u    ���        @	                       @                                                                                                           
    @        � ^��E�  ���   � ���                                             @                      I    ���        @	                       @                                                                                                           
    @        -d f� Is   ���      ��                                            Guasto 
Rimosso @                      >    ���    	   Arial @                       @                                                                                                          
    @        � � T� 8  ���     ���                                            @                      %    ���        @	        FullWarehousePuls             @                                                                                                          
    @        ]���y�  ���     ���                                            @                      &    ���        @	     	   CoverLoad             @                                                                                                           
    @        !d Z� =x   ���      ��                                 2   ExtractionCylinderSpring.PushButtonFullSpringHouse       Mag.
molle
riempito @                      +    ���    	   Arial @                       @                                                                                                           
    @        � � � � �   ���      ��                                    FullWarehousePuls       Mag. 
basi
riempito @                      ,    ���    	   Arial @                       @                                                                                                           
    @        ^h��z|  ���      ��                                 "   FullWareHouseInAssemblyStationPuls       Mag.
coperchi
riempito @                      -    ���    	   Arial @                       @                                                                                                           
    @        �d � �v   ���     ���                                            Magazzino
molle vuoto @                      /    ���    	   Arial @                       @                                                                                                           
    @        2 � w 
T �   ���     ���                                            Magazzino
basi vuoto @                      1    ���    	   Arial @                       @                                                                                                           
    @        � hI�!z  ���     ���                                            Magazzino
coperchi vuoto @                      2    ���    	   Arial @                       @                                                                                                           
    @        3 hx �U z  ���     ���                                            Pezzo
capovolto @                      3    ���    	   Arial @                       @                                                                                                           
    @        � � ;
�   ���     ���                                            Pezzo
Rosso/Met. @                      4    ���    	   Arial @                       @                                                                                                           
    @        P� �
r�   ���     ���                                            Pezzo
Nero @                      5    ���    	   Arial @                       @                                                                                                           
    @        � d ?� v   ���     ���                                            Mot. Robot
Acceso @                      9    ���    	   Arial @                       @                                                                                                           
    @        Td �� tv   ���     ���                                            Robot In
movimento @                      :    ���    	   Arial @                       @                                                                                                          
    @        � �� �� �  ���     ���                                            @                      V    ���        @	     $   UpsideDownLoadRemovedInExpellingPuls             @                                                                                                           
    @        � h� �� |  ���      ��                                 $   UpsideDownLoadRemovedInExpellingPuls       Pezzo
capovolto
rimosso @                      W    ���    	   Arial @                       @                                                                                                           
    @        � d � � � x   ���     ���                                            Pezzi neri
o Rossi/Met. @                      _    ���    	   Arial @                       @                                                                                                         
    @        < � y � Z �     @                     @���     ���             @    n    ���        @	    ToWorkBlackOrRedLoadPuls                 @       �                                                                                                     
    @        � � � � � �     @                     @���     ���             @    o    ���        @	    ToWorkBlackLoadPuls                 @       �                                                                                                      
    @        ( IQ �<    ���     ���       �                                 %   Additional buttons (virtual only) @                      ~    ���    	   Arial @                       @                                                                                                          
    @        2 � y � U �     ��     ���                                NOT ToWorkBlackOrRedLoad        	   Tutti @                      �    ���        @	                       @                                                                                                          
    @        � � � � � �            ���     ���                        NOT ToWorkBlackLoad           Neri @                      �    ���    	   Arial @	                       @                                                                                                          
    @        � � � � � �    �       ���                                ToWorkBlackLoad           Rossi/Met. @                      �    ���        @	                       @                                                                                                         
    @        2 ( � G � 7     @                 0   Enable virtual switchboard @���     ���             @    j    ���    	   Arial @
    EnableVirtualBox                 @       �                                                                                                       
    @        ( �G J7    ���     ���                                            Virtual @                      �    ���        @	                       @                                                                                                           
    @        ( �G J7    �     ���                                EnableVirtualBox           Physical @                      �    ���        @	                       @                                                                                                          
    @        2 � y � U �    ���     ���                                ToWorkBlackOrRedLoad           Solo un tipo: @                      ]    ���        @	                       @                                                                                                           
    @        < d y � Z v   ���     ���                                            Selezione
pezzi @                      �    ���    	   Arial @                       @                                                                                                           
    @        �� gc�  ���   � ���                                             @                      �    ���        @	                       @                                                                                                          
    @        \X95  ���     ��                                    EmptyPistonSilverWarehouse        @                      �    ���        @	                       @                                                                                                           
    @        �� ���   ���     ���                                            Mag. 
pistoni
riempiti @                      �    ���    	   Arial @                       @                                                                                                           
    @        	� i9�   ���     ���                                         "   Magazzino
pistoni grigi vuoto @                      �    ���    	   Arial @                       @                                                                                                          
    @        ��S�7  ���     ���                                            @                      �    ���        @	     
   PistonLoad             @                                                                                                          
    @        �<X5  ���     ��                                    EmptyPistonBlackWarehouse        @                      �    ���        @	                       @                                                                                                           
    @        �� L�   ���     ���                                         !   Magazzino
pistoni neri vuoto @                      �    ���    	   Arial @                       @                                                                                                          
    @        (� i� H�    �@     ���                                            @                      �    ���        @	        FaultDetected             @                                                                                                          
    @        �� �� ��   ���     ��                                    EmptySpringWarehouse        @                      �    ���        @	                       @                                                                                                          
    @        P� �� s�   ���     ��                                    LightRobotInMovement        @                      �    ���        @	                       @                                                                                                          
    @        � � =� �   ���     ��                                    LightRobotServoON        @                      �    ���        @	                       @                                                                                                          
    @        "� Z� >�   ���     ���                                            @                      �    ���        @	     
   SpringLoad             @                                                                                                          
    @        � ;X6  ���     �                                     LightRedLoad        @                      �    ���        @	                       @                                                                                                          
    @        P�Xs6  ���     ���                                   LightBlackLoad        @                      �    ���        @	                       @                                                                                                          
    @        2 x SU 0  ���     ��@                                   LightEmptyWarehouse        @                      �    ���        @	                       @                                                                                                          
    @        1 �y �U �  ���       �                                   LightUpsideDownLoadInExpelling        @                      �    ���        @	                       @                                                                                                          
    @        � �D� �  ���     ��                                     EmptyCoverHouseInAssemblyStation        @                      �    ���        @	                       @                                                                                                         
    @        ����0�    @                 #   Back to Plant @���     ���             @    �    ���    	   Arial @                      @    FESTO_Interface  �                                                                                                       
    @        �bg���  ���   � ���                                             @                      �    ���        @	                       @                                                                                                         
    @        	m��F�    @                 +   Fill All 
Warehouses @���     ���             @    �    ���    	   Arial @        FillAllWarehouses             @       �                                                                                                     
    @        �tR��    @                 #   Remove Pieces @���     ���             @    �    ���    	   Arial @        Remove             @       �         �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���                  0   , K�. jd           Virtual_Panel ��d
    @    ��da   d   E                                                                                                     
    @         � [ 5 �       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @          [ � 6 J       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @        a  � � � D       ���     ��� ���     I   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\PULSANTEELFIN.BMP                       �               @                     ���        @	                               ���                                                                                                      
    @         ; X  6 ]             �                                 5   ExtractionCylinderSpring.CommandLightEmptySpringHouse        @                           ���        @	                       @                                                                                                          
    @         � X � 5 �           �                                   2   RotarySelectorPistons.CommandLightEmptyPistonHouse        @                          ���        @	                       @                                                                                                     1   8
    @        �  �$    ���     ���                                     2   ExtractionCylinderSpring.NumberOfSpringInWareHouse#   Numero di molle in magazzino %s @                          ���        @	                      @                                                                                                     1   8
    @        � � �� *�   ���     ���                                     4   RotarySelectorPistons.NumberOfBlackPistonInWareHouse*   Numero di pistoni neri in magazzino %s @                          ���        @	                      @                                                                                                     1   8
    @        � � �� *�   ���     ���                                     4   RotarySelectorPistons.NumberOfWhitePistonInWareHouse-   Numero di pistoni bianchi in magazzino %s @                          ���        @	                      @                                                                                                          
    @        e E � | � `    @      ���                                             @                          ���        @	     2   ExtractionCylinderSpring.PushButtonFullSpringHouse             @                                                                                                         
    @        a � � � �       ���     ��� ���     I   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\PULSANTEELFIN.BMP                       �               @                     ���        @	                               ���                                                                                                      
    @        e � � � � �    @      ���                                             @                          ���        @	     /   RotarySelectorPistons.PushButtonFullPistonHouse             @                                                                                                         
    @         X~ 8A       ���     ��� ���     I   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\PULSANTEELFIN.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @        � ~ �F       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @        � �~ �A       ���     ��� ���     I   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\PULSANTEELFIN.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @        g �~ �F       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                     ���        @	                               ���                                                                                                     
    @        �� ��       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                 !    ���        @	                               ���                                                                                                      
    @        �6 | �Y           ��@                                    LightEmptyWarehouse        @                      "    ���        @	                       @                                                                                                          
    @        �� ��             �                                    LightUpsideDownLoadInExpelling        @                      #    ���        @	                       @                                                                                                          
    @        l7 �} �Z           ��                                     LightEmptyCoverHouse        @                      $    ���        @	                       @                                                                                                          
    @        B Ty 8]    @      ���                                             @                      %    ���        @	        PulsFullWareHouse             @                                                                                                          
    @        �B �y �]    @      ���                                             @                      '    ���        @	     "   PulsFullWareHouseInAssemblyStation             @                                                                                                         
    @        � d=�       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                 (    ���        @	                               ���                                                                                                      
    @        � `=�           �                                      LightRedLoad        @                      )    ���        @	                       @                                                                                                         
    @        l� ���       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                 *    ���        @	                               ���                                                                                                      
    @        p� ���           ���                                    LightBlackLoad        @                      +    ���        @	                       @                                                                               ColorVisualization[1]                          
    @        I (x T` >  ���     ���                                    TRUE        @                      1    ���        @	                       @                                                                               ColorVisualization[2]                          
    @        � (� T� >  ���     ���                                    TRUE        @                      2    ���        @	                       @                                                                               ColorVisualization[3]                          
    @        � )U� ?  ���     ���                                    TRUE        @                      3    ���        @	                       @                                                                               ColorVisualization[4]                          
    @        ()WU??  ���     ���                                    TRUE        @                      4    ���        @	                       @                                                                               ColorVisualization[5]                          
    @        l(�T�>  ���     ���                                    TRUE        @                      5    ���        @	                       @                                                                               PositionVisualization[1]                          
    @        � [� �� q  ���     ���                                    TRUE        @                      6    ���        @	                       @                                                                               PositionVisualization[2]                          
    @        � [�� q  ���     ���                                    TRUE        @                      7    ���        @	                       @                                                                               PositionVisualization[3]                          
    @        '[V�>q  ���     ���                                    TRUE        @                      8    ���        @	                       @                                                                               PositionVisualization[4]                          
    @        lZ���p  ���     ���                                    TRUE        @                      9    ���        @	                       @                                                                                                           
    @         ���� �  ���     ���                                         M   Inizio Giostra     Controllo        Foratura      Espulsione        Robot @                      :    ���        @	                       @                                                                                                           
    @        �0�E�:  ���     ���                                         
   Colore @                      ;    ���        @	                       @                                                                                                           
    @        �cz�n  ���     ���                                            Posizione @                      <    ���        @	                       @                                                                                                           
    @        e 
 � @ � %   ���     ���                                         )   Pulsante 
Magazzino
molle
riempito @                      >    ���    	   Arial @                       @                                                                                                           
    @         T> 7#   ���     ���                                         (   Pulsante
Magazzino 
basi
riempito @                      ?    ���    	   Arial @                       @                                                                                                           
    @        � �> �#   ���     ���                                         ,   Pulsante 
Magazzino
coperchi
riempito @                      @    ���    	   Arial @                       @                                                                                                           
    @        d � � � � �   ���     ���                                         +   Pulsante
Magazzino 
pistoni
riempito @                      A    ���    	   Arial @                       @                                                                                                           
    @          X 8 5 &   ���     ���                                            Magazzino
molle vuoto @                      B    ���    	   Arial @                       @                                                                                                           
    @         � W � 4 �   ���     ���                                            Magazzino
pistoni vuoto @                      C    ���    	   Arial @                       @                                                                                                           
    @        � 3 �!   ���     ���                                            Magazzino
basi vuoto @                      D    ���    	   Arial @                       @                                                                                                           
    @        k �4 �"   ���     ���                                            Magazzino
coperchi vuoto @                      E    ���    	   Arial @                       @                                                                                                           
    @        �� 
� ��   ���     ���                                            Pezzo
capovolto @                      F    ���    	   Arial @                       @                                                                                                           
    @        � ^� ;�   ���     ���                                            Pezzo
Rosso/Met. @                      G    ���    	   Arial @                       @                                                                                                           
    @        p� �� ��   ���     ���                                            Pezzo
Nero @                      H    ���    	   Arial @                       @                                                                                                         
    @        �� ��       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                 I    ���        @	                               ���                                                                                                     
    @        � g@�       ���     ��� ���     L   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\LAMPADAORIGINALE.BMP                       �               @                 K    ���        @	                               ���                                                                                                      
    @        � c@�           �                                      LightRobotInMovement        @                      L    ���        @	                       @                                                                                                           
    @        �� � ��   ���     ���                                            Mot. Robot
Acceso @                      M    ���    	   Arial @                       @                                                                                                           
    @        � b� ?�   ���     ���                                            Robot In
movimento @                      N    ���    	   Arial @                       @                                                                                                          
    @        �� ��           ��                                     LightRobotServoON        @                      O    ���        @	                       @                                                                                                         
    @         ��-~p      ���     ��� ���     \   Y:\MATERIALE\SISTEMILABORATORIO\FESTO\PROGETTICODESYS\MAINDANIELE\IMMAGINI\MACCHINASTATI.BMP                       �               @                 P    ���        @	                               ���                                                                                                      
    @        ? �� �� �  �        �                                     IndicatorStateIdle       IDLE @                      Q    ���        @	                       @                                                                                                          
    @        ���^�  �        �                                     IndicatorStateInit       INIT @                      R    ���        @	                       @                                                                                                          
    @         E� �\ i  �        �                                     IndicatorStateCheck    	   CHECK @                      S    ���        @	                       @                                                                                                          
    @        � ���;�  �        �                                     IndicatorStateAlarm    	   ALARM @                      T    ���        @	                       @                                                                                                          
    @        ��z�/�  �        �                                     IndicatorStateReady    	   READY @                      U    ���        @	                       @                                                                                                          
    @        �H~�3l  �        �                                     IndicatorStateBusy       BUSY @                      V    ���        @	                       @                                                                                                          
    @        ��O*  �        �                                     IndicatorStateStop    
   END_OP @                      W    ���        @	                       @                                                                                                          
    @        � &	r� L  �        �                                     IndicatorStateSafe       SAFE @                      X    ���        @	                       @                                                                                                          
    @        T��*�  �        �                                     IndicatorStateFreeze       FREEZE STATE @                      Y    ���        @	                       @                                                                                                           
    @        ,=�'   �@     �                                  ButtonMachineToBeEmptyInvisible   LightMachineToBeEmpty   TextMachineEmptyForReset   %s @                      Z    ���        @	    MachineNOTOkForReset                 @                                                                                                         
    @         H} (@       ���     ��� ���     I   D:\DOCUMENTI\DANY\TESI BOLOGNA\PROGETTICODESYS\IMMAGINI\PULSANTEELFIN.BMP                       �               @                 [    ���        @	                               ���                                                                                                      
    @        A Ex )\    @      ���                                             @                      \    ���        @	        FaultDetected             @                                                                                                           
    @         D= '"   ���     ���                                            Pulsante
Guasto 
Rimosso @                      ]    ���    	   Arial @                       @                                                                                                           
    @        [h��q   �      � �                                    FaultDetected OR PossibleFault   TypeOfFault   Fault rilevato:%s @                      ^    ���        @	        FaultRemoved             @             �   ��   �   ��   � � � ���     �   ��   �   ��   � � � ���      ����                 ����,   ��         $   STANDARD.LIB 22.11.02 12:08:30 @�7�S    IECSFC.LIB 2.6.14 10:37:46 @�7�S%   ANALYZATION.LIB 2.6.14 10:37:46 @�7�S      CONCAT @                	   CTD @        	   CTU @        
   CTUD @           DELETE @           F_TRIG @        
   FIND @           INSERT @        
   LEFT @        	   LEN @        	   MID @           R_TRIG @           REPLACE @           RIGHT @           RS @        	   RTC @        
   SEMA @           SR @        	   TOF @        	   TON @           TP @               F   SFCActionControl @      SFCActionType       SFCStepType                      Globale_Variablen @              AnalyzeExpression @                   AppendErrorString @              Globale_Variablen @                        , ��R ��           2 �  �           ����������������  
             ����, � � �	        ����, - ; \�                      POUs               Bridges               DistTest                 Bridge_Distribution_GDs  7                   Bridge_Testing_GDs  U   ����              Making                 Bridge_Drilling_GDs  8                   Bridge_Inspection_GSs  M                   Bridge_Processing_GDs  B   ����              Robot                 Bridge_Assembly_GDs  >                   Bridge_ItemsAssembling_GDs  g                   Bridge_ItemsSupply_GDs  l                   Bridge_PickandPlace_GDs  )   ����                Buttons  n                   Lights  %   ����           	   Libraries                Memory              	   Save_data  e                
   Shift_data  d                   Testing_colour  i                   Testing_orientation  :   ����                Generic_Device  Q                   Signal_Filter  ^   ����              SignalManagement                 SignalControl_PRG  Y                   SignalManagement  o   ����            
   Simulation                 PlantAssemblaggio  �                  PlantCarico  1                   PlantLavorazione  �                  PlantMagazzino  2                   PlantScarico  3                   PlantVerification  5                   Pulsantiera  �  ����              SubSystem_PRG               DistTest                 Distribution_PRG  9                   Testing_PRG  X   ����              Making               SubProcessing                 DrillingUnit_PRG  N                   InspectionUnit_PRG  I   ����                Processing_PRG  D   ����              Robot               SubAssembly                 ItemsAssembling_PRG  a                   ItemsSupply_PRG  m   ����                Assembly_PRG  u                   PickandPlace_PRG  (   ��������           
   System_PRG               Machine                 DistTest_PRG  C                
   Making_PRG  L                	   Robot_PRG  +   ����               Main_PRG  \   ��������           
   Data types               Handlers                 Data_Handler  -                   Subsystem_Handler  ,                   System_Handler  `   ����           	   Libraries                 GenericDevice_States  R                   SignalManagement_States  s   ����              SubSystem_States               DistTest                 Distribution_States  =                   Testing_States  Z   ����              Making                 DrillingUnit_States  O                   InspectionUnit_States  K                   Processing_States  E   ����              Robot                 Assembly_States  f                   ItemsAssemblingUnit_States  .                   ItemsSupplyUnit_States  j                   PickandPlace_States  '   ��������              System_States               Machine                 DisTest_States  ;                   Making_States  T                   Robot_States  &   ����               Main_States  _   ��������              Visualizations                 Fault_Detection  A                   Fault_Simulation  @                   FESTO_Interface  /               	   HMI_Panel  V                   Initial_Values  4                   Switchboard  ?                   Virtual_Panel  0   ����              Global Variables               Bridges                 Button_Bridge_Global_Variables  W                  Light_Bridge_Global_Variables  H   ����              GDs Global Variables               DistTest                 Distribution_Global_Variables  G                   Testing_Global_Variables  [   ����              Making                 DrillingUnit_Global_Variables  P                   InspectionUnit_Global_Variables  J                   Processing_Global_Variables  ]   ����              Robot                 Assembly_Global_Variables  <                    ItemsAssembling_Global_Variables  c                   ItemsSupply_Global_Variables  k                   PickandPlace_Global_Variables  *   ����                GDs_Timeout_Configuration  b   ����           	   Libraries                 GenericDevice_Global_Variables  S                   Memory_Global_Variables  h   ����              SignalManagement                 SignalControl_Global_Variables  v                %   SignalManagement_LIB_Global_Variables  w   ����            
   Simulation                 Simulation_Global_Variables  F   ����             &   Handlers_Comunication_Global_Variables  {   ����                                         ��������             ��.H               �\                	   localhost            P      	   localhost            P      	   localhost            P     �2J   <�$�