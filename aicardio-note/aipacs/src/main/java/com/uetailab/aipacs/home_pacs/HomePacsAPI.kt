/*
 * Copyright 2021 UET-AILAB
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

package com.uetailab.aipacs.home_pacs

import okhttp3.*
import okhttp3.logging.HttpLoggingInterceptor
import okio.*
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.http.*
import java.util.concurrent.TimeUnit



interface HomePacsAPI  {

    @POST("get_file_dicom")
    suspend fun getFileDicom02(@Body data: RequestBody): Call<ResponseBody>

    @GET("study__patient_case")
    suspend fun getStudyCase(
        @Query("IDStudy") idStudy: String ) : ResponseBody

    @GET("study")
    suspend fun getListStudies(@Query("TypeData") typeData: String) : ResponseBody

    @POST("get_file_zip_study_id")
    suspend fun getFileZipStudyID(@Body data: RequestBody): ResponseBody

    @POST("get_file_mp4_by_relative_path")
    suspend fun getFileMP4ByRelativePath(@Body data: RequestBody): ResponseBody



    @POST("save_data_annotation_case")
    suspend fun saveDataAnnotationCase(@Body data: RequestBody): ResponseBody


    @GET("get_data_annotation_case")
    suspend fun getDataAnnotationCase(
        @Query("StudyID") idStudy: String,
        @Query("DeviceID") deviceID: String
    ): ResponseBody



    interface ProgressDownloadListener {
        fun update(bytesRead: Long, contentLength: Long, done: Boolean)
    }


    @POST("query_cache")
    suspend fun getAutoEFRelativePath(@Body data: RequestBody): ResponseBody


    @Multipart
    @POST("query")
    suspend fun uploadStudyFileMP4(
        @Part file: MultipartBody.Part,
        @Part("metadata") metadata: RequestBody
//        @Part("siuid") siuid: String,
//        @Part("sopiuid") sopiuid: String,
//        @Part("frame_time") frame_time: Float,
//        @Part("x_scale") x_scale: Float,
//        @Part("y_scale") y_scale: Float,
//        @Part("heart_rate") heart_rate: Float,
//        @Part("window") window: Float
    ): ResponseBody

    companion object {
        const val TAG = "HomePacsAPI"
        val BASE_URL = "http://68.183.186.28:5000/api/v1/"
        val BASE_URL_NGROK = "http://118.70.181.146:6060/"
        val interceptor = HttpLoggingInterceptor().apply {
//            this.level = HttpLoggingInterceptor.Level.BODY
        }



        class MyActClient(val downloadListener: ProgressDownloadListener) {
            val client = OkHttpClient.Builder().apply {

                this.addInterceptor(interceptor)
                    .connectTimeout(120, TimeUnit.SECONDS)
                    .readTimeout(120, TimeUnit.SECONDS)
                    .writeTimeout(120, TimeUnit.SECONDS)
                    .addNetworkInterceptor { chain ->
                        val originalResponse: Response = chain.proceed(chain.request())
                        originalResponse.newBuilder()
                            .body(originalResponse.body?.let { ProgressResponseBody(it, progressDownloadListener = downloadListener) })
                            .build()

                    }.build()
            }.addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY })
                .build()
        }

        val clientNoListener = OkHttpClient.Builder().apply {
            this.addInterceptor(interceptor)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(20, TimeUnit.SECONDS)
                .writeTimeout(25, TimeUnit.SECONDS)
                .addNetworkInterceptor { chain ->
                    val originalResponse: Response = chain.proceed(chain.request())
//                    val downloadProgressListener = object : ProgressListener {
//                        override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
//                            Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
//                        }
//
//                    }
                    originalResponse.newBuilder()
//                        .body(originalResponse.body?.let { ProgressResponseBody(it, progressListener = downloadProgressListener) })
                        .build()
                }.build()}
            .addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY })
            .build()


        fun create(downloadListener: ProgressDownloadListener): HomePacsAPI {
            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(MyActClient(downloadListener).client)
                .build()
                .create(HomePacsAPI::class.java)
        }
        fun createBaseURLNgrok(downloadListener: ProgressDownloadListener, base_url: String=BASE_URL_NGROK): HomePacsAPI {
            return Retrofit.Builder()
                .baseUrl(base_url)
                .client(MyActClient(downloadListener).client)
                .build()
                .create(HomePacsAPI::class.java)
        }



        fun createNoListener(): HomePacsAPI {
            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(clientNoListener)
                .build()
                .create(HomePacsAPI::class.java)

        }
        fun createNoListenerBaseURL(base_url: String=BASE_URL_NGROK): HomePacsAPI {
            return Retrofit.Builder()
                .baseUrl(base_url)
                .client(clientNoListener)
                .build()
                .create(HomePacsAPI::class.java)

        }

    }

    class ProgressResponseBody(private val responseBody: ResponseBody, val progressDownloadListener: ProgressDownloadListener) : ResponseBody() {
        private var bufferedSource: BufferedSource? = null

        override fun contentType(): MediaType? {
            return responseBody.contentType()
        }

        override fun contentLength(): Long {
            return responseBody.contentLength()
        }

        override fun source(): BufferedSource {
            if (bufferedSource == null) {
                bufferedSource = source(responseBody.source()).buffer()
            }
            return bufferedSource!!
        }

        private fun source(source: Source): Source {
            return object: ForwardingSource(source) {
                var totalBytesRead = 0L

                override fun read(sink: Buffer, byteCount: Long): Long {
                    val bytesRead = super.read(sink, byteCount)
                    // read() returns the number of bytes read, or -1 if this source is exhausted.
                    totalBytesRead += if (bytesRead != -1L) bytesRead else 0
                    progressDownloadListener.update(totalBytesRead, responseBody.contentLength(), bytesRead == -1L)
                    return bytesRead
                }
            }
        }
    }
}
