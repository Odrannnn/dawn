// Copyright 2020 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef SRC_DAWN_NATIVE_OPENGL_QUERYSETGL_H_
#define SRC_DAWN_NATIVE_OPENGL_QUERYSETGL_H_

#include <vector>

#include "src/dawn/native/QuerySet.h"
#include "src/dawn/native/opengl/opengl_platform.h"

namespace dawn::native::opengl {

class Device;
struct OpenGLFunctions;

class QuerySet final : public QuerySetBase {
  public:
    static ResultOrError<Ref<QuerySet>> Create(Device* device,
                                               const QuerySetDescriptor* descriptor);

    // Whether timestamp results can be copied into a buffer on the GPU (GL_QUERY_BUFFER, GL 4.4
    // or GL_ARB_query_buffer_object). Without it, timestamp sets keep kTimestampGenerations sets
    // of GL queries and resolves report the newest generation whose results are all available,
    // a frame or two late, so the CPU never waits for the GPU.
    static bool CanResolveTimestampsOnGPU(const OpenGLFunctions& gl);

    GLuint Get(QueryIndex index) const;
    const std::vector<GLuint>& GetQueries() const;

    // The GL query a timestamp write at `index` goes to.
    GLuint BeginTimestampWrite(QueryIndex index);
    // Reads `count` timestamps from `first` without waiting, as described above. Queries not
    // written in the reported generation read 0; all read 0 while no generation is complete.
    MaybeError ReadTimestampsWithoutWaiting(const OpenGLFunctions& gl,
                                            QueryIndex first,
                                            QueryIndex count,
                                            uint64_t* values);

  private:
    static constexpr uint32_t kTimestampGenerations = 4;

    QuerySet(Device* device, const QuerySetDescriptor* descriptor);

    ~QuerySet() override;
    void DestroyImpl(DestroyReason reason) override;

    size_t Slot(uint32_t generation, QueryIndex index) const;

    // mGenerations sets of GetQueryCount() queries, generation-major. Sized on the GL thread.
    std::vector<GLuint> mQueries;
    std::vector<bool> mWritten;
    uint32_t mGenerations = 1;
    uint32_t mWriteGeneration = 0;
    bool mResolvedSinceWrite = false;
};

}  // namespace dawn::native::opengl

#endif  // SRC_DAWN_NATIVE_OPENGL_QUERYSETGL_H_
