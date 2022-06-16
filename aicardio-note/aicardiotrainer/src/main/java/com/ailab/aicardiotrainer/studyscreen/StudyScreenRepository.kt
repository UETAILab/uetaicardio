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

package com.ailab.aicardiotrainer.studyscreen

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

class StudyScreenRepository {

    companion object {
        const val TAG = "StudyScreenRepository"

        // For Singleton instantiation
        @Volatile
        private var instance: StudyScreenRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: StudyScreenRepository()
                        .also { instance = it }
            }
        const val KEY_SIUID = "siuid"
        const val KEY_STUDY_ID = "StudyID"
        const val KEY_STUDY_INSTANCE_UID = "StudyInstanceUID"
        const val KEY_RELATIVE_PATH = "RelativePath"

        const val KEY_PATH_FILE = "path_file"
        const val FOLDER_THUMBNAIL = ".thumbnail"
    }


    suspend fun downloadMP4File(listener: ProgressListener, studyID: String, studyInstanceUID: String, relativePath: String) : LCE<String> = withContext(Dispatchers.IO) {
        // Todo download file mp4 here
        val fileMP4Path = "/storage/emulated/0/Download/${studyID}/${relativePath}.mp4"

        val fileMP4 = File(fileMP4Path)
        if (fileMP4.exists()) {
            Log.w(TAG, "downloadMP4File ${fileMP4Path} exists")
            return@withContext LCE.Result(data = fileMP4Path, error = false, message = "success")
        } else {
            // get mp4 of study from server
            try {

                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_STUDY_ID, studyID)
                data.put(KEY_STUDY_INSTANCE_UID, studyInstanceUID)
                data.put(KEY_RELATIVE_PATH, relativePath)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = StudyApi.create(listener).getFileMP4ByRelativePath(bodyRequest)
                // sau khi lay duoc du lieu tu server -> save data to disk thanh file id_study.zip

                val saveFileResponse = saveResponseFileToDisk(response, fileMP4Path)
                if (saveFileResponse) {
                    Log.w(TAG, "downloadMP4File: --- ${fileMP4Path}")
                    return@withContext LCE.Result(data = fileMP4Path, error = false, message = "fileMP4Path")
                }

                else return@withContext LCE.Result(data = fileMP4Path, error = true, message = "error")

            } catch (e: Exception) {
                Log.w(TAG, "downloadMP4File ERROR $e")
                return@withContext LCE.Result(data = fileMP4Path, error = true, message = "error")
            }
        }


        return@withContext LCE.Result(data = fileMP4Path, error = true, message = "error")
    }
    suspend fun downloadJPGPreview(listener: ProgressListener, studyID: String, studyInstanceUID: String): LCE<DownloadStudyZipResult> = withContext(Dispatchers.IO) {
        // dau tien la xu ly download file

//        Log.w(TAG, "On downloadJPGPreview")
        val results = getFileZipStudyID(listener, studyID, studyInstanceUID)

        if (results) {

            // get .zip file success thi phai xu li unzip 2 truong hop
            // da unzip hoac chua unzip
            val downloadStudyZipResult = unzipFileZipStudy(studyID)

            return@withContext LCE.Result(data = downloadStudyZipResult, error = false, message = "success")
        }

        return@withContext LCE.Result(data = DownloadStudyZipResult(), error = true, message = "failed")
    }

    suspend fun getFileZipStudyID(listener: ProgressListener, studyID: String, studyInstanceUID: String): Boolean = withContext(Dispatchers.IO) {

        val fileZipPath = "/storage/emulated/0/Download/${studyID}/${studyID}.zip"

        val fileZip = File(fileZipPath)
        // neu file zip ton tai trong folder download, tuc la da download o lan truoc do
        // luc do thi chi viec load len thoi
        if (fileZip.exists()) {
            Log.w(TAG, "getFileZipStudy .zip exists")
            return@withContext true
            // file .zip of study exist

        } else {
            // get .zip of study from server
            try {

                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_STUDY_ID, studyID)
                data.put(KEY_STUDY_INSTANCE_UID, studyInstanceUID)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = StudyApi.create(listener).getFileZipStudyID(bodyRequest)
                // sau khi lay duoc du lieu tu server -> save data to disk thanh file id_study.zip
                return@withContext saveResponseZipToDisk(response, studyID)

            } catch (e: Exception) {
                Log.w(TAG, "getFileZipStudyID ERROR $e")
                return@withContext false
            }
        }
        return@withContext false
    }

    suspend fun saveResponseFileToDisk(response: ResponseBody, fileOutStreamPath: String): Boolean = withContext(Dispatchers.IO) {
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


    suspend fun saveResponseZipToDisk(response: ResponseBody, StudyID: String): Boolean = withContext(Dispatchers.IO) {
        var stream : InputStream? = null
        var outStream : FileOutputStream? = null
        try {
            stream = response.byteStream()
            val fileOutStream = File("/storage/emulated/0/Download/${StudyID}/${StudyID}.zip")
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


    suspend fun unzipFileZipStudy(studyID: String): DownloadStudyZipResult = withContext(Dispatchers.IO){
        val fileZipPath = "/storage/emulated/0/Download/${studyID}/${studyID}.zip"
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





    suspend fun saveResponseDicomToDisk(response: ResponseBody, studyID: String, fileName: String): String? = withContext(Dispatchers.IO) {
        var stream : InputStream? = null
        var outStream : FileOutputStream? = null
        try {
            stream = response.byteStream()
            val fileDicomLocal = "/storage/emulated/0/Download/${studyID}/${fileName}"
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




    suspend fun getFileDicom(dicomJPGPath: String, listener: ProgressListener): String? = withContext(Dispatchers.IO) {
        //sampe jpg_path: storage/emulated/0/Download/1.2.410.200010.1073940.8869.142001.2674626.2674626/.thumbnail/K3AA0MG2.jpg

        val studyValue = getDicomstudyID(dicomJPGPath)
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

            val response = StudyApi.create(listener).getFileDicom(bodyRequest)
            return@withContext saveResponseDicomToDisk(response, studyID = studyValue.first, fileName = studyValue.second)


        } catch (e: Exception) {
            Log.w(TAG, "getFileDicom ERROR $e")
            return@withContext null
        }

        return@withContext null
    }

    private fun getDicomstudyID(dicomJPGPath: String): Pair<String, String> {
        val p1 = dicomJPGPath.split("/")
        val name = p1.last()
        val length = p1.size
        return Pair(p1[length-3], name.substring(0, name.length-4))
//        return
    }

    suspend fun getInformationStudy(studyID: Int): LCE<JSONObject> = withContext(Dispatchers.IO) {
//        val data = JSONObject()
//        Log.w(TAG, "get StudyID: ${"%06d".format(studyID)}")
        try {
            val response = JSONObject(StudyApi.createNoListener().getStudyCase(idStudy = "%06d".format(studyID)).string()).getJSONObject("data")
            return@withContext LCE.Result(data = response, error = false, message = "success")

        } catch (e: Exception) {
            return@withContext LCE.Result(data = JSONObject(), error = true, message = "failed")

        }
//        return@withContext LCE.Result(data = data, error = true, message = "failed")
    }


}