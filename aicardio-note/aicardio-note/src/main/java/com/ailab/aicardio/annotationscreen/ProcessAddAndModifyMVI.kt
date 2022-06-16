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

import android.util.Log
import android.view.MotionEvent
import com.ailab.aicardio.repository.AnnotationStateEffectObject

class ProcessAddAndModifyMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: ProcessAddAndModifyMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: ProcessAddAndModifyMVI()
                        .also { instance = it }
            }
        const val TAG = "ProcessAddAndModifyUniDirectionMVI"
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

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {}

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {}

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {
            is AnnotationViewEvent.DrawPoint -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(DrawPointReducer(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.DrawBoundary -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(DrawBoundaryReducer(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.ModifyDrawPoint -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(ModifyDrawPointReducer(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.ModifyDrawBoundary -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(ModifyDrawBoundaryReducer(annotationActVM, it, annotationViewEvent))
                }
            }


        }

    }

    inner class DrawPointReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.DrawPoint)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            if (viewModel.hasNoLabel())
                return AnnotationStateEffectObject(null, AnnotationViewEffect.ShowToast("NO LABEL"))

            val obj = viewEvent.obj
            val event = obj.event
            val o = viewState.dicomAnnotation

            when(event?.action) {
                MotionEvent.ACTION_DOWN -> {
                    o.addPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
                }

                MotionEvent.ACTION_MOVE -> {
                    o.setLastPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
                }
                MotionEvent.ACTION_UP -> {
                    o.setLastPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
                    o.changeLength(frameIdx=obj.frameIdx, key=obj.key, tags=viewState.tagsDicom )
                }
                else -> {}

            }


            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.DrawPoint(viewEvent.obj.frameIdx)),
                viewModel.getRenderAnnotationFrame())
        }
    }

    inner class DrawBoundaryReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.DrawBoundary) : AnnotationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): AnnotationStateEffectObject {
            if (viewModel.hasNoLabel())
                return AnnotationStateEffectObject(null, AnnotationViewEffect.ShowToast("NO LABEL"))

            val o = viewState.dicomAnnotation
            val obj = viewEvent.obj
            val event = obj.event
            when (event?.action) {
                MotionEvent.ACTION_DOWN -> {
                    // add new path
                    o.addBoundary(obj.frameIdx, obj.ix, obj.iy, obj.key, isNewPath = true)
                }
                MotionEvent.ACTION_MOVE -> {
                    // add to current path (last path in array)
                    o.addBoundary(obj.frameIdx, obj.ix, obj.iy, obj.key, isNewPath = false)
                }
                MotionEvent.ACTION_UP -> {
                    o.addBoundary(obj.frameIdx, obj.ix, obj.iy, obj.key, isNewPath = false)
                    o.changeArea(frameIdx=obj.frameIdx, key=obj.key, tags=viewState.tagsDicom )

                }
                else -> {
                }
            }

            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.DrawBoundary(viewEvent.obj.frameIdx)),
                viewModel.getRenderAnnotationFrame())
        }
    }
    inner class ModifyDrawPointReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ModifyDrawPoint) : AnnotationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): AnnotationStateEffectObject {
            val obj = viewEvent.obj
            val event = obj.event
            val o = viewState.dicomAnnotation
            when (event?.action) {
                MotionEvent.ACTION_DOWN -> {
                    val modifyPointIndex = o.chooseModifyPoint(obj.ix, obj.iy, obj.scale, nColumn =viewState.nColumn, nRow = viewState.nRow, frameIdx = obj.frameIdx, key=obj.key )
                    Log.w(TAG, "$modifyPointIndex")
                    viewModel.setModifyPointIndex(modifyPointIndex)
                }
                MotionEvent.ACTION_MOVE -> o.moveModifyPoint(obj.modifyPointIdx, obj.ix, obj.iy, obj.frameIdx, obj.key)
                MotionEvent.ACTION_UP -> {
                    o.moveModifyPoint(obj.modifyPointIdx, obj.ix, obj.iy, obj.frameIdx, obj.key)
                    viewModel.setModifyPointIndex(-1)
                    o.changeLength(frameIdx = obj.frameIdx, key = obj.key, tags = viewState.tagsDicom)

                }
            }
            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.ModifyDrawPoint),
                viewModel.getRenderAnnotationFrame()
            )
        }
    }

    inner class ModifyDrawBoundaryReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ModifyDrawBoundary) : AnnotationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): AnnotationStateEffectObject {
            val obj = viewEvent.obj
            val event = obj.event
            val o = viewState.dicomAnnotation
            when (event?.action){
                MotionEvent.ACTION_DOWN -> {
                    val modifyBoundaryIndex = o.chooseModifyBoundary(obj.ix, obj.iy, obj.scale, nColumn = viewState.nColumn, nRow = viewState.nColumn, frameIdx = obj.frameIdx, key=obj.key)
//                Log.w("modifyBoundaryViewStateBoundary", "$modifyBoundaryIndex")
                    viewModel.setModifyBoundaryIndex(modifyBoundaryIndex)
                }
                MotionEvent.ACTION_MOVE ->{
                    o.moveModifyBoundary(obj.ix, obj.iy, bId=obj.modifyBoundaryIndex, frameIdx=obj.frameIdx, key=obj.key)
                    val modifyBoundaryIndex = o.chooseModifyBoundary(obj.ix, obj.iy, obj.scale, nColumn = viewState.nColumn, nRow = viewState.nColumn, frameIdx = obj.frameIdx, key=obj.key)
                    viewModel.setModifyBoundaryIndex(modifyBoundaryIndex)

                }
                MotionEvent.ACTION_UP ->{
                    o.moveModifyBoundary(obj.ix, obj.iy, bId=obj.modifyBoundaryIndex, frameIdx=obj.frameIdx, key=obj.key)
                    viewModel.setModifyBoundaryIndex(Pair(-1, -1))
                    o.changeArea(frameIdx=obj.frameIdx, key=obj.key, tags=viewState.tagsDicom )
                }
            }

            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.ModifyDrawBoundary),
                viewModel.getRenderAnnotationFrame())
        }

    }

}

