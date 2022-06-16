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

package com.uetailab.aipacs.home_pacs.fragment_home

import android.graphics.Bitmap
import android.util.Log
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.LCE
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewRepository.Companion.KEY_STUDY_INTERPREATION
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.ResponseBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipFile


class HomeViewRepository {
    companion object {
        const val TAG = "HomeViewRepository"

        const val KEY_STUDY_INSTANCE_UID = "StudyInstanceUID"
        const val KEY_STUDY_ID = "StudyID"
        const val KEY_RELATIVE_PATH = "RelativePath"

        const val KEY_PATH_FILE = "path_file"
        const val FOLDER_THUMBNAIL = ".thumbnail"


        // For Singleton instantiation
        @Volatile
        private var instance: HomeViewRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: HomeViewRepository()
                        .also { instance = it }
            }
    }
    suspend fun getListStudies(typeData: String="normal"): LCE<List<Int> > = withContext(Dispatchers.IO) {
        try {
//            val response = JSONObject(HomePacsAPI.createNoListener().getListStudies(typeData = typeData).string()).getJSONArray("data")
            // get case: nhoi mau co tim
            val response = JSONObject(HomePacsAPI.createNoListener().getListStudies(typeData = typeData).string()).getJSONObject("data").getJSONArray("yes")

            Log.w(TAG, "response: ${response}")
            val studies: ArrayList<String> = ArrayList()

            val results: ArrayList<Int> = ArrayList()

            repeat(response.length()) {
                studies.add(response.get(it).toString())
                val value = response.get(it).toString().trimStart('0').toInt()
                results.add(value)
            }

            Log.w(TAG, "Size: ${studies.size} ${results.size}")
            return@withContext LCE.Result(data = results, error = false, message = "success")
        } catch (e: Exception) {
            return@withContext LCE.Result(data = emptyList<Int>(), error = true, message = "failed")
        }
    }


    suspend fun getInformationStudy(studyID: Int): LCE<Pair<JSONObject, JSONObject> > = withContext(Dispatchers.IO) {
        try {
            // NOTE HERE: KEY_STUDY_INTERPREATION
            val deviceID = "0968663886"
            val responseInfoCase = JSONObject(HomePacsAPI.createNoListener().getStudyCase(idStudy = "%06d".format(studyID)).string()).getJSONObject("data")
            var responseAnnotationCase = JSONObject()
            try {
                responseAnnotationCase = JSONObject(HomePacsAPI.createNoListener().getDataAnnotationCase(idStudy = "%06d".format(studyID), deviceID = deviceID).string()).getJSONObject("data").getJSONObject(
                    KEY_STUDY_INTERPREATION)
            } catch (e: Exception) {
                Log.w(TAG, "Case ${studyID} not found data")
            }

            Log.w(TAG, "responseAnnotationCase ${responseAnnotationCase}")

            return@withContext LCE.Result(data = Pair(responseInfoCase, responseAnnotationCase), error = false, message = "success")

        } catch (e: Exception) {
            return@withContext LCE.Result(data = Pair(JSONObject(), JSONObject()), error = true, message = "failed")

        }
    }

    suspend fun downloadAndExtractMP4File(listener: HomePacsAPI.ProgressDownloadListener, studyID: String, studyInstanceUID: String, relativePath: String)
            : LCE< List<Bitmap> > = withContext(Dispatchers.IO) {
        val fileMP4Path = "/storage/emulated/0/Download/${studyID}/${relativePath}.mp4"

        val fileMP4 = File(fileMP4Path)
//        Log.w(TAG, "fileMP4: ${fileMP4Path}")
        val extrator = ExtractMPEGFrames()

        if (!fileMP4.exists()) {
            // get mp4 of study from server
            try {

                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_STUDY_ID, studyID)
                data.put(KEY_STUDY_INSTANCE_UID, studyInstanceUID)
                data.put(KEY_RELATIVE_PATH, relativePath)

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
                Log.w(TAG, "downloadMP4File ERROR $e")
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
            Log.w(TAG, "saveResponseZipToDisk done")
            return true

        }  catch (e: Exception) {
            e.printStackTrace()
            Log.w(TAG, "saveResponseZipToDisk $e")
            return false
        } finally {
            stream?.close()
            outStream?.close()
        }
        return false

    }

    suspend fun downloadAndSaveStudyPreview(listener: HomePacsAPI.ProgressDownloadListener, studyID: String, studyInstanceUID: String): LCE<List <String> > = withContext(Dispatchers.IO) {
        // dau tien la xu ly download file

//        Log.w(TAG, "On downloadJPGPreview")
        val results = getFileZipStudyID(listener, studyID, studyInstanceUID)

        if (results) {

            // get .zip file success thi phai xu li unzip 2 truong hop
            // da unzip hoac chua unzip
            val files = unzipFileZipStudy(studyID)

            return@withContext LCE.Result(data = files, error = false, message = "success")
        }

        return@withContext LCE.Result(data = emptyList<String>(), error = true, message = "failed")
    }

    suspend fun getFileZipStudyID(listener: HomePacsAPI.ProgressDownloadListener, studyID: String, studyInstanceUID: String): Boolean {

        val fileZipPath = "/storage/emulated/0/Download/${studyID}/${studyID}.zip"

        val fileZip = File(fileZipPath)
        // neu file zip ton tai trong folder download, tuc la da download o lan truoc do
        // luc do thi chi viec load len thoi
        if (fileZip.exists()) {
            Log.w(TAG, "getFileZipStudy .zip exists")
            return true
            // file .zip of study exist
        } else {
            // get .zip of study from server
            try {
                val data = JSONObject()
                // NOTE get suiuid from StudyItem
                data.put(KEY_STUDY_ID, studyID)
                data.put(KEY_STUDY_INSTANCE_UID, studyInstanceUID)

                val bodyRequest: RequestBody = data.toString().toRequestBody("application/json".toMediaTypeOrNull())

                val response = HomePacsAPI.create(listener).getFileZipStudyID(bodyRequest)
                // sau khi lay duoc du lieu tu server -> save data to disk thanh file id_study.zip

                return saveResponseFileToDisk(response, fileZipPath)


            } catch (e: Exception) {
                Log.w(TAG, "getFileZipStudyID ERROR $e")
                return false
            }
        }
        return false
    }
    fun unzipFileZipStudy(studyID: String): List<String> {
        val fileZipPath = "/storage/emulated/0/Download/${studyID}/${studyID}.zip"
        val fileZip = File(fileZipPath)
        val thumbnailFolder = File( fileZip.parentFile ,".thumbnail")
        Log.w(TAG, "thumbnailFolder: ${thumbnailFolder} exist: ${thumbnailFolder.exists()} ${thumbnailFolder.listFiles()}")
        val dicomPaths = if (thumbnailFolder.exists()) thumbnailFolder.listFiles().filter { it -> it.absolutePath.contains(".jpg") }.map {
                file -> file.absolutePath
        } else emptyList()

        val numFileDicom = dicomPaths.size
        Log.w(TAG, "unzipFileZipStudy #case: ${numFileDicom}")

        if (numFileDicom > 0) {
            // already unzip .zip file of study
//            Log.w(TAG, "numFileDicom: ${numFileDicom}")
            return dicomPaths
        } else {
            fileZip.unzip()
            return if (thumbnailFolder.exists())thumbnailFolder.listFiles().filter { it -> it.absolutePath.contains(".jpg") }.map { file -> file.absolutePath } else emptyList()
        }
    }

    data class ZipIO (val entry: ZipEntry, val output: File)

    private fun File.unzip(unzipLocationRoot: File? = null) {

//        val rootFolder = unzipLocationRoot ?: File(parentFile.absolutePath + File.separator + nameWithoutExtension)

        val rootFolder = unzipLocationRoot ?: File(parentFile.absolutePath + File.separator + FOLDER_THUMBNAIL)
        if (!rootFolder.exists()) {
            rootFolder.mkdirs()
        }

        Log.w(TAG, "File.unzip ${rootFolder}")
        try {
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
        } catch (e: Exception) {
            Log.w(TAG, "${e}")
        }

    }

}