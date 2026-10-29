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

#include "src/dawn/native/opengl/QuerySetGL.h"

#include <utility>

#include "src/dawn/common/Range.h"
#include "src/dawn/native/opengl/DeviceGL.h"
#include "src/dawn/native/opengl/UtilsGL.h"

namespace dawn::native::opengl {

// static
ResultOrError<Ref<QuerySet>> QuerySet::Create(Device* device,
                                              const QuerySetDescriptor* descriptor) {
    Ref<QuerySet> querySet = AcquireRef(new QuerySet(device, descriptor));

    if (uint32_t{querySet->GetQueryCount()} > 0) {
        DAWN_TRY(device->EnqueueGL([querySet](const OpenGLFunctions& gl) -> MaybeError {
            if (querySet->GetQueryType() == wgpu::QueryType::Timestamp &&
                !CanResolveTimestampsOnGPU(gl)) {
                querySet->mGenerations = kTimestampGenerations;
            }
            const size_t count =
                size_t{uint32_t{querySet->GetQueryCount()}} * querySet->mGenerations;
            querySet->mQueries.resize(count);
            querySet->mWritten.assign(count, false);
            DAWN_GL_TRY(gl, GenQueries(checked_cast<GLsizei>(count), querySet->mQueries.data()));
            return {};
        }));
    }

    return std::move(querySet);
}

QuerySet::QuerySet(Device* device, const QuerySetDescriptor* descriptor)
    : QuerySetBase(device, descriptor) {}

QuerySet::~QuerySet() = default;

void QuerySet::DestroyImpl(DestroyReason reason) {
    auto device = ToBackend(GetDevice());

    if (!mQueries.empty()) {
        IgnoreErrors(device->EnqueueDestroyGL(
            this, &QuerySet::GetQueries, reason,
            [](const OpenGLFunctions& gl,
               const std::vector<GLuint>& queries) -> MaybeError {
                DAWN_GL_TRY_IGNORE_ERRORS(
                    gl, DeleteQueries(checked_cast<GLsizei>(queries.size()), queries.data()));
                return {};
            }));
    }
    QuerySetBase::DestroyImpl(reason);
}

// static
bool QuerySet::CanResolveTimestampsOnGPU(const OpenGLFunctions& gl) {
    return gl.IsAtLeastGL(4, 4) || gl.IsGLExtensionSupported("GL_ARB_query_buffer_object");
}

size_t QuerySet::Slot(uint32_t generation, QueryIndex index) const {
    return size_t{generation} * uint32_t{GetQueryCount()} + uint32_t{index};
}

GLuint QuerySet::Get(QueryIndex index) const {
    return mQueries[Slot(mWriteGeneration, index)];
}

const std::vector<GLuint>& QuerySet::GetQueries() const {
    return mQueries;
}

GLuint QuerySet::BeginTimestampWrite(QueryIndex index) {
    if (mResolvedSinceWrite) {
        // The first write after a resolve starts the next frame's generation.
        mResolvedSinceWrite = false;
        mWriteGeneration = (mWriteGeneration + 1) % mGenerations;
        for (QueryIndex i : Range(GetQueryCount())) {
            mWritten[Slot(mWriteGeneration, i)] = false;
        }
    }
    mWritten[Slot(mWriteGeneration, index)] = true;
    return Get(index);
}

MaybeError QuerySet::ReadTimestampsWithoutWaiting(const OpenGLFunctions& gl,
                                                  QueryIndex first,
                                                  QueryIndex count,
                                                  uint64_t* values) {
    mResolvedSinceWrite = true;
    for (uint32_t age = 0; age < mGenerations; ++age) {
        const uint32_t generation = (mWriteGeneration + mGenerations - age) % mGenerations;
        bool complete = true;
        for (QueryIndex i : Range(count)) {
            const size_t slot = Slot(generation, first + i);
            if (!mWritten[slot]) {
                continue;
            }
            GLuint available = GL_FALSE;
            DAWN_GL_TRY(gl, GetQueryObjectuiv(mQueries[slot], GL_QUERY_RESULT_AVAILABLE,
                                              &available));
            if (available == GL_FALSE) {
                complete = false;
                break;
            }
        }
        if (!complete) {
            continue;
        }
        for (QueryIndex i : Range(count)) {
            const size_t slot = Slot(generation, first + i);
            GLuint64 value = 0;
            if (mWritten[slot]) {
                DAWN_GL_TRY(gl, GetQueryObjectui64v(mQueries[slot], GL_QUERY_RESULT, &value));
            }
            DAWN_UNSAFE_TODO(values[uint32_t{i}]) = value;
        }
        return {};
    }
    for (QueryIndex i : Range(count)) {
        DAWN_UNSAFE_TODO(values[uint32_t{i}]) = 0;
    }
    return {};
}

}  // namespace dawn::native::opengl
