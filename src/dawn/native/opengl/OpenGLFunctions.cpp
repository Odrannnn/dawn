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

#include "src/dawn/native/opengl/OpenGLFunctions.h"

#include <cctype>

namespace dawn::native::opengl {

MaybeError OpenGLFunctions::Initialize(GLGetProcProc getProc) {
    DAWN_TRY(mVersion.Initialize(getProc));
    if (mVersion.IsES()) {
#if defined(DAWN_ENABLE_BACKEND_OPENGLES)
        DAWN_TRY(LoadOpenGLESProcs(getProc, mVersion.GetMajor(), mVersion.GetMinor()));
#else
        return DAWN_INTERNAL_ERROR("The OpenGLES backend is not enabled");
#endif
    } else {
#if defined(DAWN_ENABLE_BACKEND_DESKTOP_GL)
        DAWN_TRY(LoadDesktopGLProcs(getProc, mVersion.GetMajor(), mVersion.GetMinor()));
#else
        return DAWN_INTERNAL_ERROR("The OpenGL backend is not enabled");
#endif
    }

    if (IsGLExtensionSupported("GL_KHR_parallel_shader_compile")) {
        MaxShaderCompilerThreadsKHR =
            reinterpret_cast<MaxShaderCompilerThreadsKHRProc>(
                getProc("glMaxShaderCompilerThreadsKHR"));
        if (MaxShaderCompilerThreadsKHR != nullptr) {
            MaxShaderCompilerThreadsKHR(0xffffffffu);
        }
    }

#if defined(DAWN_ENABLE_BACKEND_DESKTOP_GL)
    if (mVersion.IsDesktop()) {
        // LoadDesktopGLProcs only loads the procs of core versions up to the context version, but
        // many of the entry points it uses are also available through an ARB extension on older
        // contexts (notably 3.3, which has no compute). Load those by their core names when the
        // extension is present; drivers expose extension entry points under both names.
        if (!IsAtLeastGL(4, 2) && IsGLExtensionSupported("GL_ARB_texture_storage")) {
            TexStorage2D = reinterpret_cast<PFNGLTEXSTORAGE2DPROC>(getProc("glTexStorage2D"));
            TexStorage3D = reinterpret_cast<PFNGLTEXSTORAGE3DPROC>(getProc("glTexStorage3D"));
        }
        if (!IsAtLeastGL(4, 3) &&
            IsGLExtensionSupported("GL_ARB_texture_storage_multisample")) {
            TexStorage2DMultisample = reinterpret_cast<PFNGLTEXSTORAGE2DMULTISAMPLEPROC>(
                getProc("glTexStorage2DMultisample"));
        }
        if (!IsAtLeastGL(4, 3) && IsGLExtensionSupported("GL_ARB_copy_image")) {
            CopyImageSubData =
                reinterpret_cast<PFNGLCOPYIMAGESUBDATAPROC>(getProc("glCopyImageSubData"));
        }
        if (!IsAtLeastGL(4, 1) && IsGLExtensionSupported("GL_ARB_ES2_compatibility")) {
            DepthRangef = reinterpret_cast<PFNGLDEPTHRANGEFPROC>(getProc("glDepthRangef"));
            ClearDepthf = reinterpret_cast<PFNGLCLEARDEPTHFPROC>(getProc("glClearDepthf"));
        }
        if (!IsAtLeastGL(4, 3) && IsGLExtensionSupported("GL_ARB_texture_view")) {
            TextureView = reinterpret_cast<PFNGLTEXTUREVIEWPROC>(getProc("glTextureView"));
        }
        if (!IsAtLeastGL(4, 0) && IsGLExtensionSupported("GL_ARB_draw_indirect")) {
            DrawArraysIndirect =
                reinterpret_cast<PFNGLDRAWARRAYSINDIRECTPROC>(getProc("glDrawArraysIndirect"));
            DrawElementsIndirect = reinterpret_cast<PFNGLDRAWELEMENTSINDIRECTPROC>(
                getProc("glDrawElementsIndirect"));
        }
        if (!IsAtLeastGL(4, 0) && IsGLExtensionSupported("GL_ARB_draw_buffers_blend")) {
            BlendEquationSeparatei = reinterpret_cast<PFNGLBLENDEQUATIONSEPARATEIPROC>(
                getProc("glBlendEquationSeparatei"));
            BlendFuncSeparatei = reinterpret_cast<PFNGLBLENDFUNCSEPARATEIPROC>(
                getProc("glBlendFuncSeparatei"));
        }
        if (!IsAtLeastGL(4, 3) && IsGLExtensionSupported("GL_ARB_invalidate_subdata")) {
            InvalidateFramebuffer = reinterpret_cast<PFNGLINVALIDATEFRAMEBUFFERPROC>(
                getProc("glInvalidateFramebuffer"));
        }
        if (!IsAtLeastGL(4, 2) && IsGLExtensionSupported("GL_ARB_shader_image_load_store")) {
            BindImageTexture =
                reinterpret_cast<PFNGLBINDIMAGETEXTUREPROC>(getProc("glBindImageTexture"));
            MemoryBarrier = reinterpret_cast<PFNGLMEMORYBARRIERPROC>(getProc("glMemoryBarrier"));
        }
    }
#endif

    return {};
}

const OpenGLVersion& OpenGLFunctions::GetVersion() const {
    return mVersion;
}

bool OpenGLFunctions::IsAtLeastGL(uint32_t majorVersion, uint32_t minorVersion) const {
    return mVersion.IsDesktop() && mVersion.IsAtLeast(majorVersion, minorVersion);
}

bool OpenGLFunctions::IsAtLeastGLES(uint32_t majorVersion, uint32_t minorVersion) const {
    return mVersion.IsES() && mVersion.IsAtLeast(majorVersion, minorVersion);
}

bool OpenGLFunctions::SupportsParallelShaderCompile() const {
    return MaxShaderCompilerThreadsKHR != nullptr;
}

bool OpenGLFunctions::SupportsCompute() const {
    if (IsAtLeastGL(4, 3) || IsAtLeastGLES(3, 1)) {
        return true;
    }
    return IsGLExtensionSupported("GL_ARB_compute_shader") &&
           IsGLExtensionSupported("GL_ARB_shader_storage_buffer_object") &&
           IsGLExtensionSupported("GL_ARB_shader_image_load_store");
}

}  // namespace dawn::native::opengl
