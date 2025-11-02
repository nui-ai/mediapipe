#ifndef MEDIAPIPE_GPU_METAL_SHARED_RESOURCES_H_
#define MEDIAPIPE_GPU_METAL_SHARED_RESOURCES_H_

#import <CoreVideo/CVMetalTextureCache.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/NSObject.h>
#import <Metal/Metal.h>

#ifndef __OBJC__
#error This class must be built as Objective-C++.
#endif  // !__OBJC__

@interface MPPMetalSharedResources : NSObject {
}

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@property(readonly) id<MTLDevice> mtlDevice;
@property(readonly) id<MTLCommandQueue> mtlCommandQueue;
#if COREVIDEO_SUPPORTS_METAL
@property(readonly) CVMetalTextureCacheRef mtlTextureCache;
#endif

@end

namespace mediapipe_v01013_based {

class MetalSharedResources {
 public:
  MetalSharedResources();
  ~MetalSharedResources();
  MPPMetalSharedResources* resources() { return resources_; }

 private:
  MPPMetalSharedResources* resources_;
};

}  // namespace mediapipe_v01013_based

#endif  // MEDIAPIPE_GPU_METAL_SHARED_RESOURCES_H_
