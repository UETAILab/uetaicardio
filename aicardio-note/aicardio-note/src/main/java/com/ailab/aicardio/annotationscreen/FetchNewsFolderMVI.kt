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

package com.ailab.aicardio.annotationscreen

import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardio.LCE
import com.ailab.aicardio.R
import com.ailab.aicardio.annotationscreen.AnnotationActivity.Companion.bitmapHeart
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.DicomAnnotation
import com.ailab.aicardio.repository.DicomDiagnosis
import com.ailab.aicardio.repository.FolderRepository
import com.imebra.DataSet
import kotlinx.android.synthetic.main.activity_annotate.*
import kotlinx.coroutines.launch
import org.json.JSONObject

class FetchNewsFolderMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: FetchNewsFolderMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: FetchNewsFolderMVI()
                        .also { instance = it }
            }
        const val TAG = "FetchNewsFolderUniDirectionMVI"
        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance().process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance().renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance().renderViewEffect(annotationActivity, viewEffect)
        }
    }

    private val folderRepository = FolderRepository.getInstance()

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

        if (bitmapHeart == null) bitmapHeart = BitmapFactory.decodeResource(annotationActivity.resources, R.drawable.heart)

        when (viewState.status) {

            AnnotationViewStatus.FolderFetched -> {
                Log.w(TAG, "Folder Fetched")
                Log.w(TAG, "listFOLDER: ${viewState.folderList}")
                annotationActivity.newsRvFolderAdapter.submitList(viewState.folderList)
            }

            AnnotationViewStatus.FolderFetching -> {
                Log.w(TAG, "Folder Fetching")
                annotationActivity.iv_draw_canvas.setCustomImageBitmap(bitmapHeart)
            }
        }


    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {
            is AnnotationViewEvent.FetchNewsFolder -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(FetchNewsFolderReducerAsync(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.FolderFetchedError -> {
                annotationActVM.reduceStateEffectObject(
                    annotationActVM.viewStates().value?.copy(status = AnnotationViewStatus.FolderFetched),
                    AnnotationViewEffect.ShowToast(message = annotationViewEvent.result.message))

            }
            is AnnotationViewEvent.FolderFetchedSuccess -> {
//                Log.w(TAG, "FolderFetchedSuccess ${annotationViewEvent.result.data}")
                annotationActVM.reduceStateEffectObject(
                    annotationActVM.viewStates().value?.copy(status = AnnotationViewStatus.FolderFetched, folderList = annotationViewEvent.result.data),
                    null)
            }

        }
    }

    inner class FetchNewsFolderReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.FetchNewsFolder)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {


        override fun reduce(): AnnotationStateEffectObject {
            Log.w(TAG, "FetchNewsFolderReducerAsync ${viewEvent.folder}")
            var result : AnnotationStateEffectObject = AnnotationStateEffectObject(null, null)
            if (viewState.status == AnnotationViewStatus.FolderFetching) return result

            val folder = viewEvent.folder

            result = AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.FolderFetching, file="", folder = folder, bitmaps = emptyList(), dataset = DataSet(),
                    dicomAnnotation = DicomAnnotation(), dicomDiagnosis = DicomDiagnosis(),
                    machineAnnotation = DicomAnnotation(), machineDiagnosis = DicomDiagnosis(), tagsDicom = JSONObject()
                ),
                AnnotationViewEffect.ShowToast(message = "folder $folder fetching")
            )

            val currentSortKey = FolderRepository.SORT_TYPE_NAME_ASC

            viewModel.viewModelScope.launch {
                when (val readResult = folderRepository.getSetLatestFolderList(folder, currentSortKey)) {

                    is LCE.Result -> {
                        if (readResult.error) {
                            FetchNewsFolderMVI.process(viewModel, AnnotationViewEvent.FolderFetchedError(readResult))
                        }

                        else {
                            FetchNewsFolderMVI.process(viewModel, AnnotationViewEvent.FolderFetchedSuccess(readResult))
                        }
                    }
                }
            }

            return result
        }

    }
}