    max_dR = 0.4

    # calculate lepton isolation
    print "Classifying lepton isolation"
    isolated_types = [2, 6] # e, mu
    HF_types = [3, 7] # e, mu
    lep_feature_dict['lepIso_lep_isolated'] = len(lep_feature_dict)
    for i, lepton in enumerate(all_data):
        if i==0: continue
        if lepton[lep_feature_dict['lepIso_lep_truthType']] in isolated_types:
            all_data[i].append(1)
        elif lepton[lep_feature_dict['lepIso_lep_truthType']] in HF_types:
            all_data[i].append(0)
        else:
            all_data[i].append(-1) # not a recognized type

    print "Calculating ptcone variables"
    # calculate ptconeX, ptvarconeX, and topoetconeX, where X is 20, 30, 40
    # also calculate new ptcone features
    lep_feature_dict['lepIso_lep_calculated_ptcone20'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptcone30'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptcone40'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone20'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone30'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_calculated_ptvarcone40'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone20'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone30'] = len(lep_feature_dict)
    # lep_feature_dict['lepIso_lep_calculated_topoetcone40'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone20_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone30_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone40_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone20_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone30_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptvarcone40_squared'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone20_dR_weighted'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone30_dR_weighted'] = len(lep_feature_dict)
    lep_feature_dict['lepIso_lep_ptcone40_dR_weighted'] = len(lep_feature_dict)

    for i, lepton in enumerate(all_data):
        if i==0: continue
        if i%100==0: print i, "out of", len(all_data)-1, "leptons"
        associated_tracks = lepton[lep_feature_dict['lepIso_lep_associated_tracks']]
        lep_pt = lepton[lep_feature_dict['lepIso_lep_pt']]
        ptcone20 = 0
        ptcone30 = 0
        ptcone40 = 0
        ptvarcone20 = 0
        ptvarcone30 = 0
        ptvarcone40 = 0
        # topoetcone20 = 0
        # topoetcone30 = 0
        # topoetcone40 = 0
        ptcone20_squared = 0
        ptcone30_squared = 0
        ptcone40_squared = 0
        ptvarcone20_squared = 0
        ptvarcone30_squared = 0
        ptvarcone40_squared = 0
        ptcone20_dR_weighted = 0
        ptcone30_dR_weighted = 0
        ptcone40_dR_weighted = 0

        # sorted(associated_tracks, key=lambda l:l[track_feature_dict['lep_track_dR']])
        for j, track in enumerate(associated_tracks):

            # if j==0:
                # continue # skip track closest to lepton (its own track)

            # track selection criteria already applied in TrackObject.cxx in SusySkimMaker
            # the stuff below is if I want to calculate these things myself

                # if track[track_feature_dict['lepIso_track_pt']] < 1: continue
                # float eta = track[track_feature_dict['lepIso_track_eta']];
                # float theta = arctan(exp(-eta)) * 2;
                # if abs(track[track_feature_dict['lepIso_track_z0']] * sin(theta)) > 3 : continue
                # # Loose track critera from https://twiki.cern.ch/twiki/bin/view/AtlasProtected/TrackingCPRecsEarly2018
                # if track[track_feature_dict['lepIso_track_pt']] < 0.5: continue
                # if abs(track[track_feature_dict['lepIso_track_eta']]) > 2.5: continue
                # if track[track_feature_dict['lepIso_track_nSCTHits']] + track[track_feature_dict['lepIso_track_nPixHits']] < 7: continue
                # if track[track_feature_dict['lepIso_track_nSharedPixHits']] + track[track_feature_dict['lepIso_track_nSharedSCTHits']]/2 > 1: continue
                # if track[track_feature_dict['lepIso_track_nPixHoles']] + track[track_feature_dict['lepIso_track_nSCTHoles']] > 2: continue
                # if track[track_feature_dict['lepIso_track_nPixHoles']] > 1: continue

            dR = track[track_feature_dict['lep_track_dR']]
            track_pt = track[track_feature_dict['lepIso_track_pt']]
            if dR <= 0.2:
                ptcone20 += track_pt
                ptcone20_squared += track_pt * track_pt
                ptcone20_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 0.3:
                ptcone30 += track_pt
                ptcone30_squared += track_pt * track_pt
                ptcone30_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 0.4:
                ptcone40 += track_pt
                ptcone40_squared += track_pt * track_pt
                ptcone40_dR_weighted += track_pt * 0.2 / (dR + 0.01)
            if dR <= 10 / lep_pt:
                if dR <= 0.2:
                    ptvarcone20 += track_pt
                    ptvarcone20_squared += track_pt * track_pt
                if dR <= 0.3:
                    ptvarcone30 += track_pt
                    ptvarcone30_squared += track_pt * track_pt
                if dR <= 0.4:
                    ptvarcone40 += track_pt
                    ptvarcone40_squared += track_pt * track_pt

        all_data[i].append(ptcone20)
        all_data[i].append(ptcone30)
        all_data[i].append(ptcone40)
        all_data[i].append(ptvarcone20)
        all_data[i].append(ptvarcone30)
        all_data[i].append(ptvarcone40)
        # all_data[i].append(topoetcone20)
        # all_data[i].append(topoetcone30)
        # all_data[i].append(topoetcone40)
        all_data[i].append(ptcone20_squared)
        all_data[i].append(ptcone30_squared)
        all_data[i].append(ptcone40_squared)
        all_data[i].append(ptvarcone20_squared)
        all_data[i].append(ptvarcone30_squared)
        all_data[i].append(ptvarcone40_squared)
        all_data[i].append(ptcone20_dR_weighted)
        all_data[i].append(ptcone30_dR_weighted)
        all_data[i].append(ptcone40_dR_weighted)

