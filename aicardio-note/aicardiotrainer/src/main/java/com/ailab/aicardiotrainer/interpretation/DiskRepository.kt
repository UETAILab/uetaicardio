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

package com.ailab.aicardiotrainer.interpretation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import java.io.File

class DiskRepository {
    companion object {
        const val TAG = "DiskRepository"

        val DEFAULT_FOLDER_DOWNLOAD = "/storage/emulated/0/Download"

        val SORT_TYPE_NAME_DESC = "name_desc"
        val SORT_TYPE_TIME_DESC = "time_desc"
        val SORT_TYPE_NAME_ASC = "name_asc"
        val SORT_TYPE_TIME_ASC = "time_asc"

        const val TYPE_JPG = ".jpg"
        const val TYPE_MP4 = ".mp4"
        // For Singleton instantiation
        @Volatile
        private var instance: DiskRepository? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: DiskRepository()
                        .also { instance = it }
            }
    }



    suspend fun getRepresentationInStudyInstanceUID(studyInstanceUID: String): LCEResult<StudyInstanceUID> = withContext(Dispatchers.IO) {

//        val folderStudyInstanceUID = File("${DEFAULT_FOLDER_DOWNLOAD}/${studyInstanceUID}")
//        val files = File(DEFAULT_FOLDER_DOWNLOAD).listFiles()
//        files.forEach {
//            Log.w(TAG, "Download: ${it.absolutePath}")
//
//        }

        val folderStudyInstanceUID = File(DEFAULT_FOLDER_DOWNLOAD, studyInstanceUID)

        Log.w(TAG, "folderStudyInstanceUID Path: ${folderStudyInstanceUID.absolutePath}")

        val thumnailFolder = File( folderStudyInstanceUID.absolutePath, ".thumbnail") // ".thumbnail"
        Log.w(TAG, "folderStudyInstanceUID thumnailFolder: ${thumnailFolder.absolutePath}")

        val representationFiles: ArrayList<String> = ArrayList()

        val representationBitmap : ArrayList<Bitmap> = ArrayList()

        val hashMapBitmap: HashMap<String, Bitmap> = HashMap()

        if (thumnailFolder.exists()) {
            thumnailFolder.listFiles().forEach {
                val absPath = it.absolutePath
                if (absPath.contains(".jpg")) {
//                    val fileName = it.name
//                    Log.w(TAG, "fileName: ${fileName}")
//                    hashMapBitmap.put(fileName, BitmapFactory.decodeFile(absPath))
                    representationFiles.add(absPath)
                    val bitmap = BitmapFactory.decodeFile(absPath) //.toGray()
                    representationBitmap.add(bitmap)
                }
            }
        }

        val results = StudyInstanceUID(studyInstanceUID=studyInstanceUID, representationFiles = representationFiles, hashMapBitmap = hashMapBitmap, representationBitmap = representationBitmap)
        Log.w(TAG, "getRepresentationInStudyInstanceUID: ${representationFiles.size} ${representationFiles}")
        return@withContext LCEResult(data = results, error = false, message = "no_error")
    }

    suspend fun getFramesMP4SopInstanceUID(item: SopInstanceUIDItem): LCEResult<SopInstanceUIDObject> = withContext(Dispatchers.IO) {

        // imgPath=/storage/emulated/0/Download/1.2.40.0.13.0.11.2672.5.2013102492.1340595.20130717095716/1.2.840.113663.1500.1.341642571.3.13.20130717.101257.843____F9MGJJOS.jpg

        val imgPath = item.imgPath

        val sopInstanceUIDMP4Path = imgPath.replace(TYPE_JPG, TYPE_MP4)

        val extrator = ExtractMPEGFrames()
        val bitmaps = extrator.extractMPEGFrames(sopInstanceUIDMP4Path).map {
            it.toGray()
        }

        val results = SopInstanceUIDObject(sopIntanceUIDPath = sopInstanceUIDMP4Path, sopInstanceBitmaps = bitmaps)
        Log.w(TAG, "getFramesMP4SopInstanceUID ${bitmaps.size} ${sopInstanceUIDMP4Path}")

        return@withContext LCEResult(data = results, error = false, message = "no_error")
    }

}