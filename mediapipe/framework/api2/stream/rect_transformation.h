#ifndef MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_
#define MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_

#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/rect.pb.h"

namespace mediapipe_v01013_based::api2::builder {

// Updates @graph to scale @rect according to passed parameters.
Stream<mediapipe_v01013_based::NormalizedRect> Scale(Stream<mediapipe_v01013_based::NormalizedRect> rect,
                                        Stream<std::pair<int, int>> image_size,
                                        float scale_x_factor,
                                        float scale_y_factor,
                                        mediapipe_v01013_based::api2::builder::Graph& graph);

// Updates @graph to scale @rect according to passed parameters and make it a
// square that has the same center and rotation, and with the side of the square
// equal to the long side of the rect.
//
// TODO: consider removing after migrating to `Scale`.
Stream<mediapipe_v01013_based::NormalizedRect> ScaleAndMakeSquare(
    Stream<mediapipe_v01013_based::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, mediapipe_v01013_based::api2::builder::Graph& graph);

// Updates @graph to scale and shift vector of @rects according to parameters.
Stream<std::vector<mediapipe_v01013_based::NormalizedRect>> ScaleAndShift(
    Stream<std::vector<mediapipe_v01013_based::NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    mediapipe_v01013_based::api2::builder::Graph& graph);

// Updates @graph to scale and shift vector of @rects according to passed
// parameters and make each a square that has the same center and rotation, and
// with the side of the square equal to the long side of a particular rect.
//
// TODO: consider removing after migrating to `ScaleAndShift`.
Stream<std::vector<mediapipe_v01013_based::NormalizedRect>> ScaleAndShiftAndMakeSquareLong(
    Stream<std::vector<mediapipe_v01013_based::NormalizedRect>> rects,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    mediapipe_v01013_based::api2::builder::Graph& graph);

// Updates @graph to scale, shift @rect according to passed parameters.
Stream<mediapipe_v01013_based::NormalizedRect> ScaleAndShift(
    Stream<mediapipe_v01013_based::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    mediapipe_v01013_based::api2::builder::Graph& graph);

// Updates @graph to scale and shift @rect according to passed parameters and
// make it a square that has the same center and rotation, and with the side of
// the square equal to the long side of the rect.
//
// TODO: consider removing after migrating to `ScaleAndShift`.
Stream<mediapipe_v01013_based::NormalizedRect> ScaleAndShiftAndMakeSquareLong(
    Stream<mediapipe_v01013_based::NormalizedRect> rect,
    Stream<std::pair<int, int>> image_size, float scale_x_factor,
    float scale_y_factor, float shift_x, float shift_y,
    mediapipe_v01013_based::api2::builder::Graph& graph);

}  // namespace mediapipe_v01013_based::api2::builder

#endif  // MEDIAPIPE_FRAMEWORK_API2_STREAM_RECT_TRANSFORMATION_H_
