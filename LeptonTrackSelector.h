#ifndef SusySkimHiggsino_LeptonTrackSelector_h
#define SusySkimHiggsino_LeptonTrackSelector_h

//RootCore
#include "SusySkimMaker/BaseUser.h"

class LeptonTrackSelector : public BaseUser
{

 public:
  LeptonTrackSelector();
  ~LeptonTrackSelector() {};

  void setup(ConfigMgr*& configMgr);
  bool doAnalysis(ConfigMgr*& configMgr);
  bool passCuts(ConfigMgr*& configMgr);
  void finalize(ConfigMgr*& configMgr);


};

static const BaseUser* LeptonTrackSelector_instance __attribute__((used)) = new LeptonTrackSelector();

#endif
