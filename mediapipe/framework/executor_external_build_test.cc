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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/threadpool.h"

// IMPORTANT: DO NOT add "namespace hand_tracking_mp_lean" to this file.
// Leave this file outside the mediapipe namespace to emulate how MediaPipe
// clients implement and use a hand_tracking_mp_lean::Executor subclass.
namespace {

// NOTE: If we need to update this class, that means there is a
// backward-incompatible change in the MediaPipe API and MediaPipe clients also
// need to update their hand_tracking_mp_lean::Executor subclasses.
class MyExecutor : public hand_tracking_mp_lean::Executor {
 public:
  MyExecutor();
  ~MyExecutor() override;

  // To verify a hand_tracking_mp_lean::Executor subclass outside the mediapipe namespace
  // can override any method, override every method in the hand_tracking_mp_lean::Executor
  // interface.
  void AddTask(hand_tracking_mp_lean::TaskQueue* task_queue) override;
  void Schedule(std::function<void()> task) override;

 private:
  std::unique_ptr<hand_tracking_mp_lean::ThreadPool> thread_pool_;
};

MyExecutor::MyExecutor() {
  thread_pool_ = absl::make_unique<hand_tracking_mp_lean::ThreadPool>("my_executor", 1);
  thread_pool_->StartWorkers();
}

MyExecutor::~MyExecutor() { thread_pool_.reset(nullptr); }

void MyExecutor::AddTask(hand_tracking_mp_lean::TaskQueue* task_queue) {
  thread_pool_->Schedule([task_queue] { task_queue->RunNextTask(); });
}

void MyExecutor::Schedule(std::function<void()> task) {
  thread_pool_->Schedule(std::move(task));
}

class NoOpTaskQueue : public hand_tracking_mp_lean::TaskQueue {
 public:
  // Returns the number of times RunNextTask() was called.
  int call_count() const { return call_count_; }

 private:
  void RunNextTask() override { ++call_count_; }

  int call_count_ = 0;
};

TEST(ExecutorTest, MyExecutor) {
  NoOpTaskQueue task_queue;
  std::shared_ptr<MyExecutor> executor(new MyExecutor);
  int counter = 0;

  executor->AddTask(&task_queue);
  executor->Schedule([&counter] { ++counter; });
  executor->AddTask(&task_queue);
  executor->Schedule([&counter] { ++counter; });
  executor->AddTask(&task_queue);
  executor = nullptr;
  EXPECT_EQ(3, task_queue.call_count());
  EXPECT_EQ(2, counter);
}

}  // namespace
