#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace hand_tracking_mp_lean::api2::builder {

// Updates @graph to scale @rect according to passed parameters.
Stream<hand_tracking_mp_lean::NormalizedRect> Scale(Stream<hand_tracking_mp_lean::NormalizedRect> rect,
                                        Stream<std::pair<int, int>> image_size,
                                        float scale_x_factor,
                                        float scale_y_factor,
                                        hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to scale @rect according to passed parameters and make it a
// square that has the same center and rotation, and with the side of the square
// equal to the long side of the rect.
//
// TODO: consider removing after migrating to `Scale`.
Stream<hand_tracking_mp_lean::NormalizedRect> ScaleAndMakeSquare(
    Stream<hand_tracking_mp_lean::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to scale and shift vector of @rects according to parameters.
Stream<std::vector<hand_tracking_mp_lean::NormalizedRect>> ScaleAndShift(
    Stream<std::vector<hand_tracking_mp_lean::NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to scale and shift vector of @rects according to passed
// parameters and make each a square that has the same center and rotation, and
// with the side of the square equal to the long side of a particular rect.
//
// TODO: consider removing after migrating to `ScaleAndShift`.
Stream<std::vector<hand_tracking_mp_lean::NormalizedRect>> ScaleAndShiftAndMakeSquareLong(
    Stream<std::vector<hand_tracking_mp_lean::NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to scale, shift @rect according to passed parameters.
Stream<hand_tracking_mp_lean::NormalizedRect> ScaleAndShift(
    Stream<hand_tracking_mp_lean::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

// Updates @graph to scale and shift @rect according to passed parameters and
// make it a square that has the same center and rotation, and with the side of
// the square equal to the long side of the rect.
//
// TODO: consider removing after migrating to `ScaleAndShift`.
Stream<hand_tracking_mp_lean::NormalizedRect> ScaleAndShiftAndMakeSquareLong(
    Stream<hand_tracking_mp_lean::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    hand_tracking_mp_lean::api2::builder::Graph& graph);

}  // namespace hand_tracking_mp_lean::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_
