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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.util.Log
import android.view.MotionEvent

class InterpretationViewTouchEventMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationViewTouchEventMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationViewTouchEventMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationViewTouchEventMVI"

        fun process(interpretationViewVM: InterpretationViewVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationViewVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationViewFragment, viewState)
        }

        fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationViewFragment, viewEffect)
        }
    }

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {

    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.OnTouchEvent -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(TouchEventReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.OnClickUndoClearDataAnnotation -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(UndoClearAnnotationEventReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }




        }
    }


    inner class UndoClearAnnotationEventReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.OnClickUndoClearDataAnnotation) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            if (!viewModel.getIsValidFrameState()) return InterpretationViewObject()

            val currentInterpretationToolClick = viewModel.getCurrentInterpretationToolClick()
            val frameID = viewModel.getCurrentFrameIndex()
            val isClear = viewEvent.isClear

            when (currentInterpretationToolClick) {
                is InterpretationViewTool.OnClickDrawPoint -> {
                    viewModel.viewStates().value?.let {
                        it.dicomInterpretation.removeClearPointsBoundary(frameID, viewModel.keyDrawPoint, isClear)
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnUndoClearCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }

                }

                is InterpretationViewTool.OnClickDrawBoundary -> {
                    viewModel.viewStates().value?.let {
                        it.dicomInterpretation.removeClearPointsBoundary(frameID, viewModel.keyDrawBoundary, isClear)
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnUndoClearCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }
                }

                is InterpretationViewTool.OnClickMeasureLength -> {

                    viewModel.viewStates().value?.let {
                        it.dicomInterpretation.removeClearPointsBoundary(frameID, viewModel.keyMeasureLength, isClear)
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnUndoClearCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }


                }

                is InterpretationViewTool.OnClickMeasureArea -> {
                    viewModel.viewStates().value?.let {
                        it.dicomInterpretation.removeClearPointsBoundary(frameID, viewModel.keyMeasureArea, isClear)
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnUndoClearCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }
                }

            }
            return InterpretationViewObject()
        }
    }



    inner class TouchEventReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.OnTouchEvent) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            Log.w(TAG, "TouchEventReducer")
            if (!viewModel.getIsValidFrameState()) return InterpretationViewObject()

            val currentInterpretationToolClick = viewModel.getCurrentInterpretationToolClick()
            val isLongClicked = currentInterpretationToolClick.isLongClicked
            val touchEventObject = TouchEventObject(
                modifyBoundaryIndex = viewModel.getModifyBoundaryIndex(), modifyPointIdx = viewModel.getModifyPointIndex(),
                frameIdx = viewModel.getCurrentFrameIndex(), scale = viewEvent.view.getScale(), event = viewEvent.event, ix = viewEvent.ix, iy = viewEvent.iy, isLongClicked = isLongClicked)

            Log.w(TAG, "${viewEvent.ix} ${viewEvent.iy} ${touchEventObject}")

            when (currentInterpretationToolClick) {
                is InterpretationViewTool.OnClickDrawPoint -> {
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(TouchEventDrawReducer(viewModel, it, InterpretationViewEvent.DrawModifyPoint(touchEventObject = touchEventObject.copy(key = viewModel.keyDrawPoint))))
                    }

                }

                is InterpretationViewTool.OnClickDrawBoundary -> {
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(TouchEventDrawReducer(viewModel, it, InterpretationViewEvent.DrawModifyBoundary(touchEventObject = touchEventObject.copy(key = viewModel.keyDrawBoundary))))
                    }
                }

                is InterpretationViewTool.OnClickMeasureLength -> {
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(TouchEventDrawReducer(viewModel, it, InterpretationViewEvent.DrawModifyPoint(touchEventObject = touchEventObject.copy(key = viewModel.keyMeasureLength))))
                    }

                }

                is InterpretationViewTool.OnClickMeasureArea -> {
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(TouchEventDrawReducer(viewModel, it, InterpretationViewEvent.DrawModifyBoundary(touchEventObject = touchEventObject.copy(key = viewModel.keyMeasureArea))))
                    }
                }

            }

            return InterpretationViewObject()

        }

    }



    inner class TouchEventDrawReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {

            when (viewEvent) {

                is InterpretationViewEvent.DrawModifyPoint -> {
                    val obj = viewEvent.touchEventObject
                    val isModifyMod = obj.isLongClicked
                    val event = obj.event

                    val o = viewState.dicomInterpretation

                    if (isModifyMod) {
                        // modify point
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
//                                o.changeLength(frameIdx = obj.frameIdx, key = obj.key, tags = viewState.dicomMetaData)
                            }
                        }

                    } else {
                        // add point
                        when(event?.action) {
                            MotionEvent.ACTION_DOWN -> {
                                o.addPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
                            }

                            MotionEvent.ACTION_MOVE -> {
                                o.setLastPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
                            }
                            MotionEvent.ACTION_UP -> {
                                o.setLastPoint(obj.frameIdx, obj.ix, obj.iy, obj.key)
//                                o.changeLength(frameIdx=obj.frameIdx, key=obj.key, tags=viewState.dicomMetaData )
                            }
                        }

                    }
                    viewModel.viewStates().value?.let {
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnTouchDrawCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }

                }

                is InterpretationViewEvent.DrawModifyBoundary -> {
                    val obj = viewEvent.touchEventObject
                    val isModifyMod = obj.isLongClicked
                    val event = obj.event

                    val o = viewState.dicomInterpretation

                    if (isModifyMod) {
                        // modify boundary

                        when (event?.action){
                            MotionEvent.ACTION_DOWN -> {
                                val modifyBoundaryIndex = o.chooseModifyBoundary(obj.ix, obj.iy, obj.scale, nColumn = viewState.nColumn, nRow = viewState.nColumn, frameIdx = obj.frameIdx, key=obj.key)
//                                Log.w("modifyBoundaryViewStateBoundary", "$modifyBoundaryIndex")
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
//                                o.changeArea(frameIdx=obj.frameIdx, key=obj.key, tags=viewState.dicomMetaData )
                            }
                        }
                    } else {
                        // add boundary
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
//                                o.changeArea(frameIdx = obj.frameIdx, key = obj.key, tags = viewState.dicomMetaData)

                            }
                        }

                    }
                    viewModel.viewStates().value?.let {
                        return InterpretationViewObject(viewState = it.copy(status = InterpretationViewStatus.OnTouchDrawCanvas), viewEffect = viewModel.getRenderFragmentViewEffect())
                    }
                }

            }

            return InterpretationViewObject()

        }

    }





}
