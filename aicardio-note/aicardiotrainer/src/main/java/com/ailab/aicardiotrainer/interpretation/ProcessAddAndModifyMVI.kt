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

import android.util.Log
import android.view.MotionEvent

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
        const val TAG = "ProcessAddAndModifyMVI"
        fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            ProcessAddAndModifyMVI.getInstance().process(interpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
            ProcessAddAndModifyMVI.getInstance().renderViewState(interpretationActivity, viewState)
        }

        fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
            ProcessAddAndModifyMVI.getInstance().renderViewEffect(interpretationActivity, viewEffect)
        }
    }

    private fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {

    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {
        when (interpretationViewEvent) {
            is InterpretationViewEvent.DrawPoint -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(DrawPointReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.DrawBoundary -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(DrawBoundaryReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.ModifyDrawPoint -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(ModifyDrawPointReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.ModifyDrawBoundary -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(ModifyDrawBoundaryReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }


        }

    }

    inner class DrawPointReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.DrawPoint)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewModel.hasNoLabel())
                return StateEffectObject(null, InterpretationViewEffect.ShowToast("NO LABEL"))

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


            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.DrawPoint(viewEvent.obj.frameIdx)),
                viewModel.getRenderReadingMediaFrame())
        }
    }

    inner class DrawBoundaryReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.DrawBoundary) : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
            if (viewModel.hasNoLabel())
                return StateEffectObject(null, InterpretationViewEffect.ShowToast("NO LABEL"))

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

            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.DrawBoundary(viewEvent.obj.frameIdx)),
                viewModel.getRenderReadingMediaFrame())
        }
    }
    inner class ModifyDrawPointReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ModifyDrawPoint) : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
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
            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.ModifyDrawPoint),
                viewModel.getRenderReadingMediaFrame()
            )
        }
    }

    inner class ModifyDrawBoundaryReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ModifyDrawBoundary) : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
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

            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.ModifyDrawBoundary),
                viewModel.getRenderReadingMediaFrame())
        }

    }

}

