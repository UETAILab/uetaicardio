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


import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.api.load
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.inflate
import com.ailab.aicardiotrainer.repositories.FrameItem
import kotlinx.android.extensions.LayoutContainer
import kotlinx.android.synthetic.main.item_frame_horizontal.view.*

class FrameRvAdapter(
    private val listener: (View) -> Unit,
    private val longListener: (View) -> Boolean,
    private val isVertical: Boolean = false,
    private val annotationActVM: AnnotationActVM
) :
    ListAdapter<FrameItem, FrameRvAdapter.MyViewHolder>(
        NewsItemItemCallback()
    ) {
    companion object {
        const val TAG = "FrameRvAdapter"
        const val FOLDER_ITEM_LIST = "FolderItemList"
    }
    private var currentPosition = -1

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        return MyViewHolder(
            inflate(
                parent.context,
                if (isVertical) R.layout.item_frame_horizontal else R.layout.item_frame_horizontal,
                parent
            ), listener, longListener)
    }

    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(
            getItem(position), position,
            annotationActVM.hasEFBoundaryAndPoint(position),
            annotationActVM.hasGLSBoundaryAndPoint(position)
            )
    }

    override fun getItemCount() = currentList.size

    fun setCurrentPosition(frameIdx: Int) {
//        Log.w(TAG, "setCurrentPosition: id: ${frameIdx} cur: ${currentPosition}")
        if (currentPosition != -1 && currentPosition < itemCount) notifyItemChanged(currentPosition)
        currentPosition = frameIdx
        notifyItemChanged(frameIdx)
    }

    fun getCurrentPosition() : Int {
        return currentPosition
    }

    inner class MyViewHolder(
        override val containerView: View,
        listener: (View) -> Unit,
        longListener: (View) -> Boolean
    ) :
        RecyclerView.ViewHolder(containerView),
        LayoutContainer {

        init {
            itemView.setOnClickListener(listener)
            itemView.setOnLongClickListener(longListener)
        }

        fun bind(
            frameItem: FrameItem,
            position: Int,
            hasEFBoundary: Boolean,
            hasGLSBoundary : Boolean
        ) =
            with(itemView) {

                containerView.setBackgroundColor(if (position == currentPosition) containerView.context.getColor(R.color.app_blue_dark) else containerView.context.getColor(R.color.colorBackground))

                itemView.tag = frameItem
                tv_id.text = frameItem.index.toString()
                iv_Image.load(frameItem.bitmap)
//
                ll_ef.visibility = if(hasEFBoundary) View.VISIBLE else View.INVISIBLE
                ll_gls.visibility = if(hasGLSBoundary) View.VISIBLE else View.INVISIBLE
            }
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<FrameItem>() {
        override fun areItemsTheSame(oldItem: FrameItem, newItem: FrameItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: FrameItem, newItem: FrameItem): Boolean {
            return oldItem.index == newItem.index
        }
    }
}

