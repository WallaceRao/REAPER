// Author: Yonghui Rao (raoyonghui0630@google.com)

#include "epoch_tracker/feature_extractor.h"

#include <string>
#include <vector>

#include "epoch_tracker/fd_filter.h"
#include "epoch_tracker/lpc_analyzer.h"
#include "epoch_tracker/fft.h"

FeatureExtractor::FeatureExtractor(int sample_rate, float step_dur,
                                   float frame_dur, int pre_pad_frames) {
  _sample_rate = sample_rate;
  _step_dur = step_dur;
  _frame_dur = frame_dur;
  _pre_pad_frames = pre_pad_frames;
  for (int i = 0; i < pre_pad_frames; i ++) {
    _all_frame_features.push_back(FrameFeature());
  } 
}


int FeatureExtractor::feedSamples(std::vector<uint16_t> samples) {
  _samples.insert(_samples.end(), samples.begin(), samples.end());
  int step_samples = _step_dur * _sample_rate;
  int min_samples = (_pre_pad_frames + 1) * _step_dur * _sample_rate;
  float max_f0 = kMaxF0Search;
  float min_f0 = kMinF0Search;
  bool do_hilbert_transform = kDoHilbertTransform;
  bool do_high_pass = kDoHighpass;
  int ret = 0;
  int frames_in_step = _step_dur / _frame_dur + 1e-5;
  while(_samples.size() >= min_samples) {
    EpochTracker frame_et;
    frame_et.set_unvoiced_cost(kUnvoicedCost);
    if (!frame_et.Init((int16_t *)&_samples[0], min_samples, _sample_rate,
        min_f0, max_f0, do_high_pass, do_hilbert_transform)) {
        std::cout << "init wav epoch track failed" << std::endl;
        return false;
    }
    Track *f0 = NULL;
    Track *pm = NULL;
    Track *corr = NULL;
    float external_frame_interval = kExternalFrameInterval;
    float inter_pulse = kUnvoicedPulseInterval;
    std::vector<FrameFeature> step_frames_features;
    for (int i = 0; i< frames_in_step; i ++){
      step_frames_features.push_back(FrameFeature());
    }
    if (!compute(frame_et,
                 inter_pulse,
                 external_frame_interval,
                 step_frames_features,
                 &pm, &f0, &corr)) {
      std::cout << "compute feature failed" << std::endl;
      return false;
    }
    _latest_frame_feature = step_frames_features.back();
    //std::cout << "1 samples size now:" << _samples.size() << " min_samples:" << min_samples << std::endl;
    _samples.erase(_samples.begin(), _samples.begin() + step_samples);
    //std::cout << "2 samples size now:" << _samples.size() << " min_samples:" << min_samples << std::endl;
    _all_frame_features.insert(_all_frame_features.end(), step_frames_features.begin(), step_frames_features.end());
    delete f0;
    delete pm;
    delete corr;
    ret ++;
  }
  return ret;
}


Track* MakeF0(EpochTracker &et, float resample_interval, Track** cor, std::vector<FrameFeature> &features) {
  std::vector<float> f0;
  std::vector<float> corr;
  if (!et.ResampleAndReturnResults(resample_interval, &f0, &corr)) {
    return NULL;
  }
  float step = f0.size() * 1.0 /  features.size();
  for (int i = 0; i < features.size(); i ++) {
    int index = i * step;
    features[i].f0 = f0[index];
  }
  Track* f0_track = new Track;
  Track* cor_track = new Track;
  f0_track->resize(f0.size());
  cor_track->resize(corr.size());
  for (int32_t i = 0; i < f0.size(); ++i) {
    float t = resample_interval * i;
    f0_track->t(i) = t;
    cor_track->t(i) = t;
    f0_track->set_v(i, (f0[i] > 0.0) ? true : false);
    cor_track->set_v(i, (f0[i] > 0.0) ? true : false);
    f0_track->a(i) = (f0[i] > 0.0) ? f0[i] : -1.0;
    cor_track->a(i) = corr[i];
  }
  *cor = cor_track;
  return f0_track;
}

Track* MakeEpoch(EpochTracker &et, float unvoiced_pm_interval) {
  std::vector<float> times;
  std::vector<int16_t> voicing;
  et.GetFilledEpochs(unvoiced_pm_interval, &times, &voicing);
  Track* pm_track = new Track;
  pm_track->resize(times.size());
  for (int32_t i = 0; i < times.size(); ++i) {
    pm_track->t(i) = times[i];
    pm_track->set_v(i, voicing[i]);
  }
  return pm_track;
}

bool FeatureExtractor::compute(EpochTracker &et,
                               float unvoiced_pulse_interval,
                               float external_frame_interval,
                               std::vector<FrameFeature> &features,
                               Track** pm, Track** f0, Track** corr) {
  if (!et.ComputeFrameFeatures(features)) {
    return false;
  }
  std::cout << "ComputeFrameFeatures finished" << std::endl;
  bool tr_result = et.TrackEpochs();
  et.WriteDiagnostics("");  // Try to save them here, even after tracking failure.
  if (!tr_result) {
    fprintf(stderr, "Problems in TrackEpochs");
    return false;
  }
  // create pm and f0 objects, these need to be freed in calling client.
  *pm = MakeEpoch(et, unvoiced_pulse_interval);
  *f0 = MakeF0(et, external_frame_interval, corr, features);
  auto energys = et.get_energys();
  float energy_step = energys.size() * 1.0 / features.size();
  for (int i = 0; i < features.size(); i ++) {
    int energy_index = i * energy_step;
    features[i].energy = energys[energy_index];
  }
  // calculate db from the max value of spectral_density
  for (auto &feature: features) {
    float max_density = 0;
    for (int i = 0; i < feature.spectral_density.size(); i ++) {
      if (feature.spectral_density[i] > max_density) {
        max_density = feature.spectral_density[i];
      }
    }
    feature.db = 20.0 * log10(max_density);
  }
  return true;
}