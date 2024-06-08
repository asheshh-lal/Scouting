# schemas.py
from pydantic import BaseModel

class PlayerBase(BaseModel):
    new_clusters: str
    player: str
    nation: str
    pos: str
    squad: str
    comp: str
    age: float
    born: float
    mp: int
    starts: int
    min: str
    nineties: float
    ast_per_90: float
    npg_per_90: float
    npg_a_per_90: float
    xA_per_90: float
    npxG_per_90: float
    npxG_xA_per_90: float
    shots_per_90: float
    SoTs_per_90: float
    SoT_pct: float
    Gls_per_shot: float
    Gls_per_SoT: float
    AvgShotDist: float
    FKSht_per_90: float
    npxG_per_Shot: float
    np_G_xG: float
    PassCmp_per_90: float
    PassAtt_per_90: float
    PassCmp_pct: float
    TotDistPass_per_90: float
    PrgDistPass_per_90: float
    ShortCmp_per_90: float
    ShortAtt_per_90: float
    ShortCmp_pct: float
    MedCmp_per_90: float
    MedAtt_per_90: float
    MedCmp_pct: float
    LongCmp_per_90: float
    LongAtt_per_90: float
    LongCmp_pct: float
    KeyPass_per_90: float
    PassIntoThird_per_90: float
    PassIntoBox_per_90: float
    CrossIntoBox_per_90: float
    ProgPass_per_90: float
    LivePassAtt_per_90: float
    DeadPassAtt_per_90: float
    FKPassAtt_per_90: float
    TBCmp_per_90: float
    PassUnderPress_per_90: float
    Switches_per_90: float
    Crosses_per_90: float
    GroundPass_per_90: float
    LowPass_per_90: float
    HighPass_per_90: float
    LeftPass_per_90: float
    RightPass_per_90: float
    HeadPass_per_90: float
    ThrowPass_per_90: float
    OtherPartPass_per_90: float
    OffsidePass_per_90: float
    OutOBPass_per_90: float
    PassesInt_per_90: float
    PassesBlk_per_90: float
    SCA_per_90: float
    PassLiveSCA_per_90: float
    PassDeadSCA_per_90: float
    DribSCA_per_90: float
    ShSCA_per_90: float
    FoulSCA_per_90: float
    DefSCA_per_90: float
    GCA_per_90: float
    PassLiveGCA_per_90: float
    PassDeadGCA_per_90: float
    DribGCA_per_90: float
    ShGCA_per_90: float
    FoulGCA_per_90: float
    DefGCA_per_90: float
    TklAtt_per_90: float
    TklW_per_90: float
    Def_3rdTkl_per_90: float
    Mid_3rdTkl_per_90: float
    Att_3rdTkl_per_90: float
    TklvDribW_per_90: float
    TklvDribAtt_per_90: float
    Tkl_pct_v_Drib: float
    DribPast_per_90: float
    PressAtt_per_90: float
    SuccPress_per_90: float
    PressSucc_pct: float
    Def_3rdPress_per_90: float
    Mid_3rdPress_per_90: float
    Att_3rdPress_per_90: float
    Blocks_per_90: float
    ShotBlocks_per_90: float
    PassBlk_per_90: float
    Interceptions_per_90: float
    Clearances_per_90: float
    ErrToShot_per_90: float
    Touches_per_90: float
    DefPenTchs_per_90: float
    Def_3rdTchs_per_90: float
    Mid_3rdTchs_per_90: float
    Att_3rdTchs_per_90: float
    AttPenTchs_per_90: float
    TchsDefPen_pct: float
    TchsDefThrd_pct: float
    TchsMidThrd_pct: float
    TchsAttThrd_pct: float
    TchsAttPen_pct: float
    LiveTchs_per_90: float
    SuccDrib_per_90: float
    AttDrib_per_90: float
    DribSucc_pct: float
    PlayersDribPast_per_90: float
    Megs_per_90: float
    Carries_per_90: float
    TotDistCarry_per_90: float
    PrgDistCarry_per_90: float
    ProgCarry_per_90: float
    CarryIntoThird_per_90: float
    CarryIntoBox_per_90: float
    Miscontrol_per_90: float
    Dispossessed_per_90: float
    PassTarget_per_90: float
    PassesReceived_per_90: float
    PassRec_pct: float
    ProgPassReceived_per_90: float
