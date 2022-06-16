/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.api

import okhttp3.Interceptor
import okhttp3.Response
// https://blog.playmoweb.com/view-download-progress-on-android-using-retrofit2-and-okhttp3-83ed704cb968

class DownloadProgressInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val originalResponse = chain.proceed(chain.request())
        val responseBuilder = originalResponse.newBuilder()


//        val downloadProgressListener = object : DownloadProgressListener {
//            override fun update(downloadIdentifier: String, bytesRead: Long, contentLength: Long, done: Boolean) {
//                progressEventBus.post(ProgressEvent(downloadIdentifier, contentLength, bytesRead))
//            }
//        }
//
//        val downloadResponseBody = DownloadProgressResponseBody(downloadIdentifier!!, originalResponse.body()!!, downloadProgressListener)
//
//        responseBuilder.body(downloadResponseBody)


        return responseBuilder.build()
    }
}