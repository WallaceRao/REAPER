// Author: Yonghui Rao (raoyonghui0630@google.com)

#ifndef _FEATURE_EXTRACTOR_H_
#define _FEATURE_EXTRACTOR_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <epoch_tracker/epoch_tracker.h>
#include "core/file_resource.h"
#include "core/track.h"


class FeatureExtractor {
 public:
  // step_dur must be integral multiple of frame_dur, and it should be no less that 0.5 second
  // to get accurate result.
  // pre_pad_frames * frame_dur must be less than step_dur
  // the frame whose db is less that min_db will be treated as unvoice frame.
  FeatureExtractor(int sample_rate = 24000, float step_dur = 0.3,
                  float frame_dur = 0.01, int pre_pad_frames = 20,
                  float min_db = 35.0);
  ~FeatureExtractor() {};
  void reset() {
    _samples.clear();
    _all_frame_features.clear();
    _latest_frame_feature = FrameFeature();
  }

  int feedSamples(std::vector<int16_t> samples, bool is_last = false);
  bool compute(EpochTracker &et,
               float unvoiced_pulse_interval,
               float external_frame_interval,
               std::vector<FrameFeature> &features,
               Track** pm, Track** f0, Track** corr);

  FrameFeature getLatestFeature() {
    return _latest_frame_feature;
  }

  std::vector<FrameFeature> getAllFeature() {
    return _all_frame_features;
  }

 private:
  int _sample_rate = 24000;
  float _step_dur = 0.2;
  float _min_db = 40.0;
  float _frame_dur = 0.025;
  int _pre_pad_frames = 0;
  std::vector<int16_t> _samples;
  FrameFeature _latest_frame_feature;
  std::vector<FrameFeature> _all_frame_features;
  std::vector<int16_t> _pad_samples;
};

#endif  // _FFT_H_
