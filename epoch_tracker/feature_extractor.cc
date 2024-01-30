// Author: Yonghui Rao (raoyonghui0630@google.com)

#include "epoch_tracker/feature_extractor.h"

#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

#include "epoch_tracker/fd_filter.h"
#include "epoch_tracker/lpc_analyzer.h"
#include "epoch_tracker/fft.h"

FeatureExtractor::FeatureExtractor(int sample_rate, float step_dur,
                                   float frame_dur, int pre_pad_frames,
                                   float min_db) {
  _sample_rate = sample_rate;
  _step_dur = step_dur;
  _frame_dur = frame_dur;
  _pre_pad_frames = pre_pad_frames;
  _min_db = min_db;
  if (pre_pad_frames * frame_dur > step_dur) {
    std::cerr << "invalid pre_pad_frames:" << pre_pad_frames << std::endl;
  }
  int pad_samples_size = 1.0 * _pre_pad_frames * frame_dur * sample_rate + 1e-1;
  _pad_samples.clear();
  for (int i = 0; i < pad_samples_size; i ++) {
    _pad_samples.push_back(0);
  }
}


int FeatureExtractor::feedSamples(std::vector<int16_t> samples, bool is_last) {
  _samples.insert(_samples.end(), samples.begin(), samples.end());
  if (_samples.empty()) {
    return 0;
  }
  int pad_samples_size = _pad_samples.size();
  int step_samples = _step_dur * _sample_rate;
  int min_samples =  _step_dur * _sample_rate;
  float max_f0 = kMaxF0Search;
  float min_f0 = kMinF0Search;
  bool do_hilbert_transform = kDoHilbertTransform;
  bool do_high_pass = kDoHighpass;
  int ret = 0;
  int frames_in_step = _step_dur / _frame_dur + 1e-5;
  //std::cout << "is last:" << is_last << std::endl;
  if (is_last) {
    EpochTracker frame_et;
    frame_et.set_unvoiced_cost(kUnvoicedCost);
    auto samples_with_pad = _pad_samples;
    samples_with_pad.insert(samples_with_pad.end(), _samples.begin(), _samples.end());
    std::cout << "samples_with_pad size:" << samples_with_pad.size() << std::endl;
    if (!frame_et.Init((int16_t *)&samples_with_pad[0], samples_with_pad.size(), _sample_rate,
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
    int frames_count = 1.0 * samples_with_pad.size() / _sample_rate  / _frame_dur + 1e-5;
    std::cout << "last frame count:" << frames_count << ", sample count:" << samples_with_pad.size() << ", _pad_samples size:" << _pad_samples.size() << std::endl;
    for (int i = 0; i< frames_count; i ++){
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
    // compute dBs
    int samples_step = samples_with_pad.size() * 1.0 / step_frames_features.size();
    for (int i = 0; i < step_frames_features.size(); i ++) {
      int start_idx = i * samples_step;
      int end_idx = std::min(start_idx + samples_step, (int)samples_with_pad.size());
      int sum_val = 0;
      for (int j = start_idx; j < end_idx; j ++) {
          sum_val += abs(samples_with_pad[j]);
      }
      if (sum_val > 0) {
        step_frames_features[i].db = std::max(0.0, 20.0 * log10(1.0 * sum_val / (end_idx - start_idx)));
      } else {
        step_frames_features[i].db = 0;
      }
      if (step_frames_features[i].db < _min_db) {
        step_frames_features[i].f0 = 0.0;
      }
    }
    _latest_frame_feature = step_frames_features.back();
    // update _pad_samples
    memcpy(&_pad_samples[0], &_samples[_samples.size() - pad_samples_size], pad_samples_size * sizeof(int16_t));
    _samples.clear(); 
    _all_frame_features.insert(_all_frame_features.end(), step_frames_features.begin() + _pre_pad_frames, step_frames_features.end());
    delete f0;
    delete pm;
    delete corr;
    ret += step_frames_features.size() - _pre_pad_frames;
  } else {
    while(_samples.size() >= min_samples) {
      EpochTracker frame_et;
      frame_et.set_unvoiced_cost(kUnvoicedCost);
      auto samples_with_pad = _pad_samples;
      samples_with_pad.insert(samples_with_pad.end(), _samples.begin(), _samples.begin() + min_samples);
      //std::cout << "samples_with_pad size:" << samples_with_pad.size() << std::endl;
      if (!frame_et.Init((int16_t *)&samples_with_pad[0], samples_with_pad.size(), _sample_rate,
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
      for (int i = 0; i< frames_in_step + _pre_pad_frames; i ++){
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
      // compute dBs
      int samples_step = samples_with_pad.size() * 1.0 / step_frames_features.size();
      for (int i = 0; i < step_frames_features.size(); i ++) {
        int start_idx = i * samples_step;
        int end_idx = std::min(start_idx + samples_step, (int)samples_with_pad.size());
        int sum_val = 0;
        for (int j = start_idx; j < end_idx; j ++) {
            sum_val += abs(samples_with_pad[j]);
        }
        if (sum_val > 0) {
          step_frames_features[i].db = std::max(0.0, 20.0 * log10(1.0 * sum_val / (end_idx - start_idx)));
        } else {
          step_frames_features[i].db = 0;
        }
        if (step_frames_features[i].db < _min_db) {
          step_frames_features[i].f0 = 0.0;
        }
      }
      // update _pad_samples
      memcpy(&_pad_samples[0], &_samples[step_samples - pad_samples_size], pad_samples_size * sizeof(int16_t));
      _samples.erase(_samples.begin(), _samples.begin() + step_samples);
      // std::cout << "step_samples:" << step_samples << ", pad_samples_size:" << pad_samples_size << std::endl;
      // std::cout << "step_frames_features size:" << step_frames_features.size() << std::endl;
      // std::cout << "_all_frame_features size:" << _all_frame_features.size() << std::endl;
      _all_frame_features.insert(_all_frame_features.end(), step_frames_features.begin() + _pre_pad_frames, step_frames_features.end());
      delete f0;
      delete pm;
      delete corr;
      ret += step_frames_features.size() - _pre_pad_frames;
    }
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
    /*
    // mute energy whose f0 is 0
    if (features[i].f0 <= 0) {
      features[i].energy = 0;
    }
    */
  }
  /*
  // calculate db from the max value of spectral_density, not
  for (auto &feature: features) {
    float max_density = 0;
    for (int i = 0; i < feature.spectral_density.size(); i ++) {
      if (feature.spectral_density[i] > max_density) {
        max_density = feature.spectral_density[i];
      }
    }
    if (max_density > 0) {
      feature.db = 20.0 * log10(max_density);
    } else {
      feature.db = 0;
    }
  }
  */
  return true;
}