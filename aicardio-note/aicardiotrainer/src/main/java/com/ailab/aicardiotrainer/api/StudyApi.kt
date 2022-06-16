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

import android.util.Log
import okhttp3.*
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.http.*
import java.util.concurrent.TimeUnit

interface StudyApi {

//    @Headers( "Content-Type: application/json; charset=utf-8")
    @POST("get_file_dicom")
    suspend fun getFileDicom(@Body data: RequestBody): ResponseBody

    @POST("get_file_zip_study")
    suspend fun getFileZipStudy(@Body data: RequestBody): ResponseBody


    @POST("get_file_zip_study_id")
    suspend fun getFileZipStudyID(@Body data: RequestBody): ResponseBody


    @POST("get_file_mp4_by_relative_path")
    suspend fun getFileMP4ByRelativePath(@Body data: RequestBody): ResponseBody


    @POST("getZipStudyInstanceUID")
    suspend fun getZipStudyInstanceUID(@Body data: RequestBody): ResponseBody


    @POST("get_file_dicom")
    suspend fun getFileDicom02(@Body data: RequestBody): Call<ResponseBody>
//    Call<Products> getUserDetails(@Query("email") String emailID, @Query("password") String password)

    @GET("study__patient_case")
    suspend fun getStudyCase(
//        @Header("Authorization") token: String,
        @Query("IDStudy") idStudy: String ) : ResponseBody

//    Call<ResponseBody> getUserDetails(@Query("IDStudy") emailID: String)


    // https://www.youtube.com/watch?v=FYBrHCgTvRQ
    // https://github.com/AnushkaMadusanka/RetrofitDemo/blob/master/app/src/main/java/com/anushka/retrofitdemo/RetrofitInstance.kt
        // https://stackoverflow.com/questions/33338181/is-it-possible-to-show-progress-bar-when-upload-image-via-retrofit-2
    // https://medium.com/impulsiveweb/retrofit-multiple-file-upload-with-progress-in-android-82634b494df3
        companion object {

            val BASE_URL = "http://68.183.186.28:5000/api/v1/"
            const val TAG = "StudyApi"
            val interceptor = HttpLoggingInterceptor().apply {
    //            this.level = HttpLoggingInterceptor.Level.BODY
            }

            class MyActClient(val listener: ProgressListener) {

                val client = OkHttpClient.Builder().apply {
                    this.addInterceptor(interceptor)
                        .connectTimeout(30, TimeUnit.SECONDS)
                        .readTimeout(20, TimeUnit.SECONDS)
                        .writeTimeout(25, TimeUnit.SECONDS)
                        .addNetworkInterceptor { chain ->
                            val originalResponse: Response = chain.proceed(chain.request())
    //                        val downloadProgressListener = object : ProgressListener {
    //                            override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
    //                                Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
    //                            }
    //
    //                        }
                            originalResponse.newBuilder()
                                .body(originalResponse.body?.let { ProgressResponseBody(it, progressListener = listener) })
                                .build()

                        }.build()
                }.build()

            }

            val clientNoListener = OkHttpClient.Builder().apply {
                this.addInterceptor(interceptor)
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .readTimeout(20, TimeUnit.SECONDS)
                    .writeTimeout(25, TimeUnit.SECONDS)
                    .addNetworkInterceptor { chain ->
                        val originalResponse: Response = chain.proceed(chain.request())
                            val downloadProgressListener = object : ProgressListener {
                                override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
                                    Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
                                }

                            }
                        originalResponse.newBuilder()
                            .body(originalResponse.body?.let { ProgressResponseBody(it, progressListener = downloadProgressListener) })
                            .build()

                    }.build()
            }.addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }).build()

            fun create(listener: ProgressListener): StudyApi {
                return Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(MyActClient(listener).client)
                    .build()
                    .create(StudyApi::class.java)
            }

            fun createNoListener(): StudyApi {
                return Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(clientNoListener)
                    .build()
                    .create(StudyApi::class.java)

            }

    }

}