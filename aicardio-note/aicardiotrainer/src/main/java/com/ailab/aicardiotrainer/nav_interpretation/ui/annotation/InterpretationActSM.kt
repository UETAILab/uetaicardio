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

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import android.graphics.Bitmap
import com.ailab.aicardiotrainer.studyscreen.DicomItem
import org.json.JSONObject

data class InterpretationViewState(
    val status: InterpretationViewStatus,

    // data for study
    val studyInstanceUIDPath: String,
    val studyFiles: List<String>,
    val studyRepresentation: List<Bitmap>,
    val studyHashMap: HashMap<String, Bitmap>,

    // data for file dicom
    val sopInstanceUIDPath: String,
    val sopInstanceUIDBitmaps: List<Bitmap>,
    val sopInstanceUIDTags: JSONObject
) {
    companion object {
        const val newWidthBitmap = 50
        const val newHeightBitmap = 35

    }
    fun getListSopInstanceUIDItem(): List<SopInstanceUIDItem> {
        return  studyFiles.mapIndexed { index, s ->
//            SopInstanceUIDItem(id = index.toLong(), name = filename(s), imgPath = s, bitmap = getResizedBitmap(studyRepresentation.get(index), newWidthBitmap, newHeightBitmap) )
            SopInstanceUIDItem(id = index.toLong(), name = filename(s), imgPath = s, bitmap = studyRepresentation.get(index) )
        }
    }

    fun filename(fullPath: String): String { // gets filename without extension
        // "/storage/emulated/0/Download/StudyInstanceUID/SOPInstanceUID____fileName.jpg"
        val dot: Int = fullPath.lastIndexOf("/")
        val sep: Int = fullPath.lastIndexOf(".jpg")
        return fullPath.substring(dot + 1, sep)
    }

    fun getShortSopInstanceUID(): String {
        val shortNameArr = sopInstanceUIDPath.split("____")
        val result = shortNameArr[shortNameArr.size - 1]

        if (result.contains(".")) {
            val sep: Int = result.lastIndexOf(".")
            return result.substring(0, sep)
        }

        return shortNameArr[shortNameArr.size - 1]
    }

    fun getSubmitListFrameItem(): List<FrameItem> {
        return this.sopInstanceUIDBitmaps.mapIndexed { index, bitmap ->
            FrameItem(index=index, bitmap = bitmap)
        }
    }


}

sealed class InterpretationViewStatus {

    object Start: InterpretationViewStatus()

//    data class OnDownloadingStudyInstanceUID(val studyInstanceUID: String): InterpretationViewStatus()

//    object OnDoneDownloadingStudyInstanceUID: InterpretationViewStatus()
//    object OnDownloadingStudyInstanceUID: InterpretationViewStatus()

    object OnLoadingRepresentationStudyInstanceUID: InterpretationViewStatus()
    object OnDoneLoadingRepresentationStudyInstanceUID: InterpretationViewStatus()

    object OnLoadingMP4SopInstanceUID: InterpretationViewStatus()
    object OnDoneLoadingMP4SopInstanceUID: InterpretationViewStatus()
    object EnhanceContrastBitmap : InterpretationViewStatus()


}

sealed class InterpretationViewEffect {
    data class ShowToast(val message: String) : InterpretationViewEffect()
    data class RenderMP4Frame(val rmf: RenderMP4FrameObject) : InterpretationViewEffect()
    data class RenderComponentActivity(val idComponent: Int, val isPlaying: Boolean) : InterpretationViewEffect()
    data class RenderTextViewGammaCorrection(val gamma: Double): InterpretationViewEffect()


}
sealed class InterpretationViewEvent {
    data class LoadingMP4File(val filePath: String) : InterpretationViewEvent()

    object NextFrame : InterpretationViewEvent()
    object ShowNextFrame : InterpretationViewEvent()

    object ShowPreviousFrame : InterpretationViewEvent()
    object ShowFirstFrame : InterpretationViewEvent()
    object ShowLastFrame : InterpretationViewEvent()
    object PlayPauseVideo : InterpretationViewEvent()

    object PlayBackMP4File: InterpretationViewEvent()

    data class LoadingRepresentationStudyInstanceUID(val studyInstanceUID: String) : InterpretationViewEvent()
    data class LoadingMP4SopInstanceUID(val item: SopInstanceUIDItem) : InterpretationViewEvent()
    data class SopInstanceUIDFrameClicked(val frameItem: FrameItem) : InterpretationViewEvent()
    data class SopInstanceUIDFrameLongClicked(val frameItem: FrameItem) : InterpretationViewEvent()
    data class EnhanceContrastBitmap(val threshold: Int) : InterpretationViewEvent()
    data class OnChangeGammaCorrection(val threshold: Int) : InterpretationViewEvent()
}

