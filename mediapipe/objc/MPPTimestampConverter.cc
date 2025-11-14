// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "MPPTimestampConverter.h"

#include <cstdint>

@implementation MPPTimestampConverter {
  hand_tracking_mp_lean::Timestamp _mediapipeTimestamp;
  hand_tracking_mp_lean::Timestamp _lastTimestamp;
  hand_tracking_mp_lean::TimestampDiff _timestampOffset;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    [self reset];
  }
  return self;
}

- (void)reset {
  _mediapipeTimestamp = hand_tracking_mp_lean::Timestamp::Min();
  _lastTimestamp = _mediapipeTimestamp;
  _timestampOffset = 0;
}

- (hand_tracking_mp_lean::Timestamp)timestampForMediaTime:(CMTime)mediaTime {
  Float64 sampleSeconds =
      CMTIME_IS_VALID(mediaTime) ? CMTimeGetSeconds(mediaTime) : 0;
  const int64_t sampleUsec =
      sampleSeconds * hand_tracking_mp_lean::Timestamp::kTimestampUnitsPerSecond;
  _mediapipeTimestamp = hand_tracking_mp_lean::Timestamp(sampleUsec) + _timestampOffset;
  if (_mediapipeTimestamp <= _lastTimestamp) {
    _timestampOffset =
        _timestampOffset + _lastTimestamp + 1 - _mediapipeTimestamp;
    _mediapipeTimestamp = _lastTimestamp + 1;
  }
  _lastTimestamp = _mediapipeTimestamp;
  return _mediapipeTimestamp;
}

@end
