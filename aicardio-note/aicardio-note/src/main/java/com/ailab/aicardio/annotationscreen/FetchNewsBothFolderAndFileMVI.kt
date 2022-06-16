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

import com.ailab.aicardio.repository.AnnotationStateEffectObject

class FetchNewsBothFolderAndFileMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: FetchNewsBothFolderAndFileMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: FetchNewsBothFolderAndFileMVI()
                        .also { instance = it }
            }
        const val TAG = "FetchNewsBothFolderAndFileUniDirectionMVI"

        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance()
                .process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance()
                .renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance()
                .renderViewEffect(annotationActivity, viewEffect)
        }
    }

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {
            is AnnotationViewEvent.FetchNewsBothFolderAndFile -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(FetchNewsBothFolderAndFileReducer(annotationActVM, it, annotationViewEvent ))

                }
            }
        }
    }





    inner class FetchNewsBothFolderAndFileReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.FetchNewsBothFolderAndFile)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            var result: AnnotationStateEffectObject = AnnotationStateEffectObject(null, null)
            if (viewState.status == AnnotationViewStatus.FolderFetching) return result
            if (viewState.status == AnnotationViewStatus.Fetching) return result

            val folder = viewEvent.folder


            val file = viewEvent.file

            result = result.copy(annotationViewEffect = AnnotationViewEffect.ShowToast(message = "folder $folder & file $file fetching"))

            folder?.let {
//                if (folder != viewState.folder)
                FetchNewsFolderMVI.process(viewModel, AnnotationViewEvent.FetchNewsFolder(folder = it))
            }

            file?.let {
//                if (file != viewEvent.file)
                FetchNewsFileMVI.process(viewModel, AnnotationViewEvent.FetchNewsFile(file = it))
            }

            return result

        }

    }



}