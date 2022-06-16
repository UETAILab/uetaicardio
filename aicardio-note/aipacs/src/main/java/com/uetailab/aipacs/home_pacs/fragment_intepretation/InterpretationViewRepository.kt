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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.graphics.Bitmap
import android.util.Log
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.LCE
import com.uetailab.aipacs.home_pacs.fragment_home.ExtractMPEGFrames
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.ResponseBody
import org.json.JSONObject
import retrofit2.http.Part
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream


class InterpretationViewRepository {
    companion object {
        const val TAG = "InterpretationViewRepository"
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationViewRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationViewRepository()
                        .also { instance = it }
            }
        const val KEY_DEVICE_ID = "DeviceID"
        const val KEY_STUDY_ID = "StudyID"
        const val KEY_STUDY_INSTANCE_UID = "StudyInstanceUID"
        const val KEY_STUDY_RELATIVE_PATH = "SopInstanceUID"
        const val KEY_DICOM_INTERPREATION = "DicomInterpretation"
        const val KEY_STUDY_INTERPREATION = "StudyInterpretation"
        const val KEY_APP_VERSION = "APP_VERSION"
        const val APP_VERSION = "5.0"
    }


    suspend fun saveAnnotationToSever(viewState: InterpretationViewState): LCE<JSONObject> = withContext(Dispatchers.IO) {

        try {
            val data = JSONObject()

            data.put(KEY_DEVICE_ID, viewState.deviceID)
            data.put(KEY_STUDY_ID, viewState.studyID)
//            data.put(KEY_DICOM_INTERPREATION, viewState.dicomInterpretation)
            data.put(KEY_STUDY_INTERPREATION, viewState.studyInterpretation)

            data.put(KEY_APP_VERSION, APP_VERSION)
            val dataRequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())
            val response = JSONObject(HomePacsAPI.createNoListener().saveDataAnnotationCase(dataRequestBody).string())

            return@withContext LCE.Result(data = response, error = false, message = "save to server done")

        } catch (e: Exception) {
            Log.w(HomeViewRepository.TAG, "saveAnnotationToSever ERROR $e")
            return@withContext LCE.Result(data = JSONObject(), error = true, message = e.toString())
        }


    }


    suspend fun getAutoEFForRelativePath(listener: HomePacsAPI.ProgressDownloadListener, studyInstanceUID: String, relativePath: String): LCE<JSONObject> = withContext(Dispatchers.IO) {
        try {
            val data = JSONObject()
            data.put(KEY_STUDY_INSTANCE_UID, studyInstanceUID)
            data.put(KEY_STUDY_RELATIVE_PATH, relativePath)

            val metadataRequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())
            val response = JSONObject(HomePacsAPI.createBaseURLNgrok(listener).getAutoEFRelativePath(metadataRequestBody).string())

            return@withContext LCE.Result(data = response, error = false, message = "error")

        } catch (e: Exception) {
            Log.w(HomeViewRepository.TAG, "getAutoEFFromJSONFile ERROR $e")
            return@withContext LCE.Result(data = JSONObject(), error = true, message = "error")
        }

        return@withContext LCE.Result(data = JSONObject(), error = true, message = "error")
    }


    suspend fun uploadStudyMP4File(listener: HomePacsAPI.ProgressDownloadListener, studyID: String, relativePath: String, metadata: JSONObject): LCE<JSONObject> = withContext(Dispatchers.IO) {

//        val result = getFileJSONFromResources(R.)

//        return@withContext LCE.Result(data = JSONObject(), error = false, message = "error")

        val fileMP4Path = "/storage/emulated/0/Download/${studyID}/${relativePath}.mp4"
        val fileMP4 = File(fileMP4Path)
        if (fileMP4.exists()) {
            try {
                val requestFile: RequestBody = fileMP4.asRequestBody("multipart/form-data".toMediaTypeOrNull())
                val multipartBody = MultipartBody.Part.createFormData("file", fileMP4Path, requestFile)
                val metadataRequestBody = metadata.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = JSONObject(HomePacsAPI.createBaseURLNgrok(listener).uploadStudyFileMP4(multipartBody, metadataRequestBody).string())
//                val response = JSONObject(HomePacsAPI.createBaseURLNgrok(listener).uploadStudyFileMP4(
//                    multipartBody, siuid, sopiuid, frame_time, x_scale, y_scale, heart_rate, window).toString())
                return@withContext LCE.Result(data = response, error = false, message = "error")

            } catch (e: Exception) {
                Log.w(HomeViewRepository.TAG, "uploadStudyMP4File ERROR $e")
                return@withContext LCE.Result(data = JSONObject(), error = true, message = "error")
            }

        }

        return@withContext LCE.Result(data = JSONObject(), error = true, message = "error")
    }

    suspend fun downloadAndExtractMP4File(listener: HomePacsAPI.ProgressDownloadListener, studyID: String, studyInstanceUID: String, relativePath: String)
            : LCE<List<Bitmap>> = withContext(Dispatchers.IO) {
        val fileMP4Path = "/storage/emulated/0/Download/${studyID}/${relativePath}.mp4"

        val fileMP4 = File(fileMP4Path)
//        Log.w(TAG, "fileMP4: ${fileMP4Path}")
        val extrator = ExtractMPEGFrames()

        if (!fileMP4.exists()) {
            // get mp4 of study from server
            try {

                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(HomeViewRepository.KEY_STUDY_ID, studyID)
                data.put(HomeViewRepository.KEY_STUDY_INSTANCE_UID, studyInstanceUID)
                data.put(HomeViewRepository.KEY_RELATIVE_PATH, relativePath)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = HomePacsAPI.create(listener).getFileMP4ByRelativePath(bodyRequest)
                // sau khi lay duoc du lieu tu server -> save data to disk thanh file id_study.zip

                val saveFileResponse = saveResponseFileToDisk(response, fileMP4Path)
//                Log.w(TAG, "saveFileResponse: ${saveFileResponse}")
                if (saveFileResponse) {
                    return@withContext LCE.Result(data = extrator.extractMPEGFrames(fileMP4Path), error = false, message = "success download file mp4 with relative path")
                }

                else return@withContext LCE.Result(data = emptyList<Bitmap>(), error = true, message = "error")

            } catch (e: Exception) {
                Log.w(HomeViewRepository.TAG, "downloadMP4File ERROR $e")
                return@withContext LCE.Result(data = emptyList<Bitmap>(), error = true, message = "error")
            }
        } else {
            // already download mp4 file
            return@withContext LCE.Result(data = extrator.extractMPEGFrames(fileMP4Path), error = false, message = "success")

        }

    }

    fun saveResponseFileToDisk(response: ResponseBody, fileOutStreamPath: String): Boolean{
        var stream : InputStream? = null
        var outStream : FileOutputStream? = null
        try {
            stream = response.byteStream()
            val fileOutStream = File(fileOutStreamPath)
            if (!fileOutStream.exists()) {
                fileOutStream.parentFile.mkdirs()
            }
            outStream = FileOutputStream(fileOutStream)
            outStream.use { out -> out.write(stream.readBytes()) }
            Log.w(HomeViewRepository.TAG, "saveResponseZipToDisk done")
            return true

        }  catch (e: Exception) {
            e.printStackTrace()
            Log.w(HomeViewRepository.TAG, "saveResponseZipToDisk $e")
            return false
        } finally {
            stream?.close()
            outStream?.close()
        }
        return false

    }
}