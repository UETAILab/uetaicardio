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

import android.graphics.Bitmap
import android.util.Log
import com.ailab.aicardiotrainer.nav_interpretation.ui.annotation.InterpretationViewEvent
import org.json.JSONObject
import java.io.File

data class StudyViewState(
    val status: StudyViewStatus,
    val skillName: String,
    val studyName: String,
    val studyId: String,
    val studyPaths: List<String> = emptyList(),
    val studyInformation: JSONObject,
    val currentFileMP4Path: String?,
    val currentBitmap: List<Bitmap> = emptyList()
) {
    companion object {
        const val TAG = "StudyViewState"
    }
    fun getListDicomItem(): List<DicomItem> {
        return studyPaths.mapIndexed { index, s ->
            DicomItem(id = index.toLong(), name = filename(s), imgPath = s )
        }
    }
    fun filename(fullPath: String): String { // gets filename without extension
        val dot: Int = fullPath.lastIndexOf("/")
        val sep: Int = fullPath.lastIndexOf(".jpg")
//        Log.w(TAG, "${dot + 1}, ${fullPath.substring(dot + 1, sep)}")
        return fullPath.substring(dot + 1, sep)
    }

    fun getStudyInstanceUID(): String? {
//        if (studyInformation.has("StudyInstanceUID")) {
//            return studyInformation.getString("StudyInstanceUID")
//        }
//        return ""
        Log.w(TAG, "${studyInformation}")
        return if (studyInformation.has("StudyInstanceUID")) studyInformation.getString("StudyInstanceUID") else null
    }

//    fun getStudyId(): String? {
//        Log.w(TAG, "${studyInformation}")
//        return if (studyInformation.has("IDStudy")) studyInformation.getString("StudyInstanceUID") else null
//    }
}

sealed class StudyViewEffect {
    data class ShowToast(val message: String) : StudyViewEffect()
}

sealed class StudyViewEvent {
    data class DownloadJPGPreview(val studyId: String, val studyInstanceUID: String) : StudyViewEvent()
    data class GetInformationStudy(val studyId: Int) : StudyViewEvent()
    data class DownloadAndExtractMP4File(val studyId: String, val studyInstanceUID: String, val relativePath: String) : StudyViewEvent()
    data class ExtractMP4File(val fileMP4Path: String) : StudyViewEvent()

}

sealed class StudyViewStatus {
    object Start : StudyViewStatus()
    object DownloadingJPGPreview : StudyViewStatus()
    object DownloadedJPGPreview : StudyViewStatus()

    object DownloadingMP4File : StudyViewStatus()
    object DownloadedMP4File : StudyViewStatus()


    object LoadedStudyInformation : StudyViewStatus()
    object ExtractedMP4File : StudyViewStatus()
    object ExtractingMP4File : StudyViewStatus()

}