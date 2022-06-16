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
import org.json.JSONObject

data class HomeViewState(
    val status: HomeViewStatus,
    val message: String,
    val studies: List<Int>,
    val studyID: String? = null,
    val studyInstanceUID: String? = null,
    val studyMetaData: JSONObject,
    val studyInterpretation: JSONObject,
    val studyPreview: List<String>,
    val relativePath: String?=null,
    val bitmaps : List<Bitmap>
) {
    fun getListStudyGVItem(): List<StudyGVItem> {
        return studyPreview.mapIndexed { index, s ->
            StudyGVItem(id = index.toLong(), name = filename(s), img_path = s )
        }
    }
    fun filename(fullPath: String): String { // gets filename without extension
        val dot: Int = fullPath.lastIndexOf("/")
        val sep: Int = fullPath.lastIndexOf(".jpg")
        return fullPath.substring(dot + 1, sep)
    }


}

sealed class HomeViewEffect {
    data class ShowToast(val message: String) : HomeViewEffect()

}
sealed class HomeViewEvent {
    data class FetchStudies(val typeData: String="normal") : HomeViewEvent()
    data class FetchStudy(val studyID: Int) : HomeViewEvent()
    data class FetchPreviewStudy(val studyID: String, val studyInstanceUID: String) : HomeViewEvent()
    data class FetchFileMP4(val studyID: String, val studyInstanceUID: String, val relativePath: String) : HomeViewEvent()

}

sealed class HomeViewStatus {
    object Start : HomeViewStatus()
    data class FetchedData(val viewEvent: HomeViewEvent) : HomeViewStatus()
    data class FetchedErrorData(val viewEvent: HomeViewEvent) : HomeViewStatus()
    data class OnFetchingData(val viewEvent: HomeViewEvent) : HomeViewStatus()
}