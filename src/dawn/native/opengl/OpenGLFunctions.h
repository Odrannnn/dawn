// Copyright 2019 The Dawn & Tint Authors
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

#ifndef SRC_DAWN_NATIVE_OPENGL_OPENGLFUNCTIONS_H_
#define SRC_DAWN_NATIVE_OPENGL_OPENGLFUNCTIONS_H_

#include <string>

#include "dawn/native/opengl/OpenGLFunctionsBase_autogen.h"
#include "src/dawn/native/opengl/OpenGLVersion.h"

#ifndef GL_MAX_SHADER_COMPILER_THREADS_KHR
#define GL_MAX_SHADER_COMPILER_THREADS_KHR 0x91B0
#endif

#ifndef GL_COMPLETION_STATUS_KHR
#define GL_COMPLETION_STATUS_KHR 0x91B1
#endif

namespace dawn::native::opengl {

using MaxShaderCompilerThreadsKHRProc = void(KHRONOS_APIENTRY*)(GLuint count);

struct OpenGLFunctions : OpenGLFunctionsBase {
  public:
    // capDesktopAtGL33 treats a desktop context of any newer version as 3.3 (GLForceDesktopGL33).
    MaybeError Initialize(GLGetProcProc getProc, bool capDesktopAtGL33 = false);

    const OpenGLVersion& GetVersion() const;
    bool IsAtLeastGL(uint32_t majorVersion, uint32_t minorVersion) const;
    bool IsAtLeastGLES(uint32_t majorVersion, uint32_t minorVersion) const;
    bool SupportsParallelShaderCompile() const;

    // Whether compute shaders, and the storage resources they need, can be used on this context.
    // A desktop OpenGL 3.3 context without the relevant extensions is graphics-only: limits for
    // compute and storage resources are reported as 0 and compute-based emulation is disabled.
    bool SupportsCompute() const;

    MaxShaderCompilerThreadsKHRProc MaxShaderCompilerThreadsKHR = nullptr;

  private:
    OpenGLVersion mVersion;
    bool mCappedAtGL33 = false;
};

}  // namespace dawn::native::opengl

#endif  // SRC_DAWN_NATIVE_OPENGL_OPENGLFUNCTIONS_H_
