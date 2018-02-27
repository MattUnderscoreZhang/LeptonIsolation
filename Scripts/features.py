lep_features = [
    ("lep_q", "int"),
    ("lep_pt", "float"),
    ("lep_eta", "float"),
    ("lep_phi", "float"),
    ("lep_m", "float"),
    ("lep_d0", "float"),
    ("lep_z0", "float"),
    ("lep_d0Err", "float"),
    ("lep_z0Err", "float"),
    ("lep_pTErr", "float"),
    ("lep_ptcone20", "float"),
    ("lep_ptcone30", "float"),
    ("lep_ptcone40", "float"),
    ("lep_topoetcone20", "float"),
    ("lep_topoetcone30", "float"),
    ("lep_topoetcone40", "float"),
    ("lep_ptvarcone20", "float"),
    ("lep_ptvarcone30", "float"),
    ("lep_ptvarcone40", "float"),
    ("lep_truthType", "int")]

# truthType info
isolatedTypes = [2, 6] # e, mu
HFTypes = [3, 7] # e, mu

track_features = [
    ("track_q", "float"),
    ("track_pt", "float"),
    ("track_eta", "float"),
    ("track_phi", "float"),
    ("track_m", "float"),
    ("track_fitQuality", "float"),
    ("track_d0", "float"),
    ("track_z0", "float"),
    ("track_d0Err", "float"),
    ("track_z0Err", "float"),
    ("track_nIBLHits", "int"),
    ("track_nPixHits", "int"),
    ("track_nPixHoles", "int"),
    ("track_nPixOutliers", "int"),
    ("track_nSCTHits", "int"),
    ("track_nTRTHits", "int")]

all_features = lep_features + track_features
