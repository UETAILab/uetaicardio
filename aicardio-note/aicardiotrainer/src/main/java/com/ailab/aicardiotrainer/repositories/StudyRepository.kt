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

package com.ailab.aicardiotrainer.repositories

import android.util.Log
import com.ailab.aicardiotrainer.DownloadStudyZipResult
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.api.ProgressListener
import com.ailab.aicardiotrainer.api.StudyApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

class StudyRepository {

    companion object {
        const val TAG = "StudyRepository"

        // For Singleton instantiation
        @Volatile
        private var instance: StudyRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: StudyRepository()
                        .also { instance = it }
            }
        const val KEY_SIUID = "siuid"
        const val KEY_StudyInstanceUID = "StudyInstanceUID"
        const val KEY_PATH_FILE = "path_file"
        const val FOLDER_THUMBNAIL = ".thumbnail"
    }
//
//    suspend fun getFileDicom(skillName: String, study: StudyItem) = withContext(Dispatchers.IO) {
//        try {
//            val data = JSONObject()
////            data.put("path_file", "abc")
////            data.put("get", "abc")
////            data.put()
//            val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())
//            val response = StudyApi.create().getFileDicom(bodyRequest)
////            saveResponseToDisk(response, skillName, study)
//        } catch (e: Exception) {
//            Log.w(TAG, "getFileDicom $e")
//        }
//    }


    suspend fun unzipFileZipStudy(studyName: String): DownloadStudyZipResult = withContext(Dispatchers.IO){
        val fileZipPath = "/storage/emulated/0/Download/${studyName}/${studyName}.zip"
        val fileZip = File(fileZipPath)
        val thumnailFolder = File( fileZip.parentFile ,".thumbnail")
        val dicomPaths = if (thumnailFolder.exists())thumnailFolder.listFiles().filter { it -> it.absolutePath.contains(".jpg") }.map {
            file -> file.absolutePath
        } else emptyList()

        val numFileDicom = dicomPaths.size
        Log.w(TAG, "unzipFileZipStudy #case: ${numFileDicom}")

        if (numFileDicom > 0) {
            // already unzip .zip file of study
            Log.w(TAG, "numFileDicom: ${numFileDicom}")
            return@withContext DownloadStudyZipResult(localPath = fileZipPath, filesPath = dicomPaths)
        } else {
            fileZip.unzip()
            return@withContext DownloadStudyZipResult( localPath = fileZipPath, filesPath =  if (thumnailFolder.exists())thumnailFolder.listFiles().filter { it -> it.absolutePath.contains(".jpg") }.map { file -> file.absolutePath
            } else emptyList() )
        }
    }

    suspend fun getFileZipStudy(studyName: String): Boolean = withContext(Dispatchers.IO) {

        val fileZipPath = "/storage/emulated/0/Download/${studyName}/${studyName}.zip"

        val fileZip = File(fileZipPath)

        if (fileZip.exists()) {
            Log.w(TAG, "getFileZipStudy .zip exists")
            return@withContext true
            // file .zip of study exist

        } else {
            // get .zip of study from server
            try {
                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_SIUID, studyName)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())
                val response = StudyApi.createNoListener().getFileZipStudy(bodyRequest)

                return@withContext saveResponseZipToDisk(response, studyName)

            } catch (e: Exception) {
                Log.w(TAG, "getFileZipStudy ERROR $e")
                return@withContext false
            }
        }
        return@withContext false
    }

    suspend fun getFileZipStudyInstanceUID(StudyInstanceUID: String): Boolean = withContext(Dispatchers.IO) {

        val fileZipPath = "/storage/emulated/0/Download/${StudyInstanceUID}/${StudyInstanceUID}.zip"

        val fileZip = File(fileZipPath)

        if (fileZip.exists()) {
            Log.w(TAG, "getFileZipStudy .zip exists")
            return@withContext true
            // file .zip of study exist

        } else {
            // get .zip of study from server
            try {

                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_StudyInstanceUID, StudyInstanceUID)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = StudyApi.createNoListener().getZipStudyInstanceUID(bodyRequest)

                return@withContext saveResponseZipToDisk(response, StudyInstanceUID)

            } catch (e: Exception) {
                Log.w(TAG, "getFileZipStudyInstanceUID ERROR $e")
                return@withContext false
            }
        }
        return@withContext false
    }



    suspend fun saveResponseZipToDisk(response: ResponseBody, StudyInstanceUID: String): Boolean = withContext(Dispatchers.IO) {
        var stream : InputStream? = null
        var outStream : FileOutputStream? = null
        try {
            stream = response.byteStream()
            val fileOutStream = File("/storage/emulated/0/Download/${StudyInstanceUID}/${StudyInstanceUID}.zip")
            if (!fileOutStream.exists()) {
                fileOutStream.parentFile.mkdirs()
            }
            outStream = FileOutputStream(fileOutStream)
            outStream.use { out -> out.write(stream.readBytes()) }
            Log.w(TAG, "saveResponseZipToDisk done")
            return@withContext true

        }  catch (e: Exception) {
            e.printStackTrace()
            Log.w(TAG, "saveResponseZipToDisk $e")

        } finally {
            stream?.close()
            outStream?.close()
        }
        return@withContext false

    }


    suspend fun saveResponseDicomToDisk(response: ResponseBody, studyName: String, fileName: String): String? = withContext(Dispatchers.IO) {
        var stream : InputStream? = null
        var outStream : FileOutputStream? = null
        try {
            stream = response.byteStream()
            val fileDicomLocal = "/storage/emulated/0/Download/${studyName}/${fileName}"
            val fileOutStream = File(fileDicomLocal)
            if (!fileOutStream.exists()) {
                fileOutStream.parentFile.mkdirs()
            }
            outStream = FileOutputStream(fileOutStream)
            outStream.use { out ->
                val streamValue = stream.readBytes()
                Log.w("TAG", "SIZE: ${streamValue.size}")
                out.write(streamValue)
            }
            Log.w(TAG, "saveResponseDicomToDisk done")
            return@withContext fileDicomLocal
        }  catch (e: Exception) {
            e.printStackTrace()
            Log.w(TAG, "saveResponseDicomToDisk $e")
        } finally {
            stream?.close()
            outStream?.close()
        }
        return@withContext null

    }


    data class ZipIO (val entry: ZipEntry, val output: File)

    private fun File.unzip(unzipLocationRoot: File? = null) {

//        val rootFolder = unzipLocationRoot ?: File(parentFile.absolutePath + File.separator + nameWithoutExtension)

        val rootFolder = unzipLocationRoot ?: File(parentFile.absolutePath + File.separator + FOLDER_THUMBNAIL)
        if (!rootFolder.exists()) {
            rootFolder.mkdirs()
        }

        Log.w(TAG, "File.unzip ${rootFolder}")

        ZipFile(this).use { zip ->
            zip
                .entries()
                .asSequence()
                .map {
                    val outputFile = File(rootFolder.absolutePath + File.separator + it.name)
                    ZipIO(it, outputFile)
                }
                .map {
                    it.output.parentFile?.run{
                        if (!exists()) mkdirs()
                    }
                    it
                }
                .filter { !it.entry.isDirectory }
                .forEach { (entry, output) ->
                    zip.getInputStream(entry).use { input ->
                        output.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
        }
    }



    suspend fun downloadDicomPreviews(studyName: String) : LCE<DownloadStudyZipResult> = withContext(Dispatchers.IO) {
        // NOTE change study name here
        val results = getFileZipStudy(studyName)

        if (results) {
            // get .zip file success
            val downloadStudyZipResult = unzipFileZipStudy(studyName)
            return@withContext LCE.Result(data = downloadStudyZipResult, error = false, message = "success")
        }

        return@withContext LCE.Result(data = DownloadStudyZipResult(), error = true, message = "failed")

    }

    suspend fun downloadDicomPreviewJPGs(StudyInstanceUID: String): LCE<DownloadStudyZipResult> = withContext(Dispatchers.IO) {
        Log.w(TAG, "On downloadDicomPreviewJPGs")
        val results = getFileZipStudyInstanceUID(StudyInstanceUID)

        if (results) {
            // get .zip file success
            val downloadStudyZipResult = unzipFileZipStudy(StudyInstanceUID)

            return@withContext LCE.Result(data = downloadStudyZipResult, error = false, message = "success")
        }

        return@withContext LCE.Result(data = DownloadStudyZipResult(), error = true, message = "failed")
    }

    suspend fun getFileDicom(dicomJPGPath: String, listener: ProgressListener): String? = withContext(Dispatchers.IO) {
        //sampe jpg_path: storage/emulated/0/Download/1.2.410.200010.1073940.8869.142001.2674626.2674626/.thumbnail/K3AA0MG2.jpg
        val studyValue = getDicomStudyName(dicomJPGPath)
        val pathDicomLocal : String = "/storage/emulated/0/Download/${studyValue.first}/${studyValue.second}"
        val dicomFile = File(pathDicomLocal)

        if (dicomFile.exists() && dicomFile.length() > 0) {
            Log.w(TAG, "File: ${pathDicomLocal} ${dicomFile.length()} exist, no download process")
            return@withContext pathDicomLocal
        }
        try {
            val data = JSONObject()
            // NOTE get suiuid from StudyItem
            data.put(KEY_SIUID, studyValue.first)
            data.put(KEY_PATH_FILE, studyValue.second)
            Log.w(TAG, "getFileDicom $data" )

            val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())
//            val res02 = StudyApi.create().getFileDicom02(bodyRequest)
//                res02.execute()
//            res02.execute()?.let {
//                Log.w(TAG, "${it.body()?.contentLength()}")
//            }

            val response = StudyApi.create(listener).getFileDicom(bodyRequest)
            return@withContext saveResponseDicomToDisk(response, studyName = studyValue.first, fileName = studyValue.second)


        } catch (e: Exception) {
            Log.w(TAG, "getFileDicom ERROR $e")
            return@withContext null
        }

        return@withContext null
    }

    private fun getDicomStudyName(dicomJPGPath: String): Pair<String, String> {
        val p1 = dicomJPGPath.split("/")
        val name = p1.last()
        val length = p1.size
        return Pair(p1[length-3], name.substring(0, name.length-4))
//        return
    }

    suspend fun downloadDicomFile(dicomJPGPath: String, listener: ProgressListener) : LCE<DicomObject> = withContext(Dispatchers.IO) {
        // NOTE change study name here
        val results = getFileDicom(dicomJPGPath, listener)

        results?.let {
            return@withContext LCE.Result(data = DicomObject(dicomPath = results ), error = false, message = "success")
        }?:run {
            return@withContext LCE.Result(data = DicomObject(), error = true, message = "failed")
        }

    }




}