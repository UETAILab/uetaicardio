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

package com.ailab.aicardiotrainer.annotationscreen

import android.content.Context
import android.graphics.Rect
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import androidx.lifecycle.viewModelScope
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.api.ProgressListener
import com.ailab.aicardiotrainer.repositories.StudyRepository
import kotlinx.coroutines.launch

class DownloadDicomFileMVI(val listener: ProgressListener) {

//    companion object {
//        // For Singleton instantiation
//        @Volatile
//        private var instance: DownloadDicomFileMVI? = null
//
//        fun getInstance() =
//            instance ?: synchronized(this) {
//                instance
//                    ?: DownloadDicomFileMVI()
//                        .also { instance = it }
//            }
//
//        const val TAG = "DownloadDicomFileMVI"
//        fun process(viewModel: AnnotationActVM, viewEvent: AnnotationViewEvent) {
//            getInstance().process(viewModel, viewEvent)
//        }
//
//        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
//            getInstance()
//                .renderViewState(annotationActivity, viewState)
//        }
//
//        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
//            getInstance()
//                .renderViewEffect(annotationActivity, viewEffect)
//        }
//
//    }

    private val studyRepository = StudyRepository.getInstance()

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
    }

    fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
        when (viewState.status) {
            AnnotationViewStatus.DownloadingDicom -> {
                annotationActivity.openProgressDialog()
            }
            AnnotationViewStatus.DownloadedDicom -> {
                annotationActivity.closeProgressDialog()
            }
        }
    }

    fun process(viewModel: AnnotationActVM, viewEvent: AnnotationViewEvent) {
        when (viewEvent) {
            is AnnotationViewEvent.DownloadDicom -> {
                viewModel.viewStates().value?.let {
                    viewModel.reduce(DownloadDicomReducer(viewModel, viewState = it, viewEvent = viewEvent))
                }
            }
        }
    }

    inner class DownloadDicomReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.DownloadDicom)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): AnnotationStateEffectObject {

            if (viewState.status is AnnotationViewStatus.DownloadingDicom) return AnnotationStateEffectObject()

            viewModel.viewModelScope.launch {

                when (val result = studyRepository.downloadDicomFile(dicomJPGPath=viewEvent.dicomJPGPath, listener =  listener)) {
                    is LCE.Result -> {
                        if (result.error) {
                            viewModel.reduce(
                                AnnotationStateEffectObject(
                                viewState.copy(status = AnnotationViewStatus.DownloadedDicom),
                                viewEffect = AnnotationViewEffect.ShowToast(message = "Error loading dicom file")
                            )
                            )

                        } else { // download success
                            // TODO process result.data
                            // reading file dicom

//                            Log.w(TAG, "${result.data}")
                            viewModel.viewStates().value?.let {
                                viewModel.reduce(AnnotationStateEffectObject(
                                    it.copy(status = AnnotationViewStatus.DownloadedDicom, file = result.data.dicomPath),
                                    viewEffect = AnnotationViewEffect.ShowToast(message = "Loading dicom done ${result.data.dicomPath}")
                                ))
                            }

                            ReadDicomFileMVI.process(viewModel, AnnotationViewEvent.FetchNewsFile(file = result.data.dicomPath))
//                            viewModel.
                            // reading Dicom

                        }
                    }
                }

            }

            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.DownloadingDicom),
                viewEffect = AnnotationViewEffect.ShowToast(message = "On Downloading dicom")
            )
        }

    }

}