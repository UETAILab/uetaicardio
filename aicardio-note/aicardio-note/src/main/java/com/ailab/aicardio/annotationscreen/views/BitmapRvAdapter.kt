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

package com.ailab.aicardio.annotationscreen.views


import android.graphics.Bitmap
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.api.load
import com.ailab.aicardio.R
import com.ailab.aicardio.inflate
import kotlinx.android.extensions.LayoutContainer
import kotlinx.android.synthetic.main.item_frame_horizontal.view.*

class BitmapRvAdapter(private val listener: (View) -> Unit, private val longListener: (View) -> Boolean, private val isVertical: Boolean = false) :
    ListAdapter<Bitmap, BitmapRvAdapter.MyViewHolder>(
        NewsItemItemCallback()
    ) {
    companion object {
        const val TAG = "BitmapRvAdapter"
    }
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        return MyViewHolder(
            inflate(
                parent.context,
                if (isVertical) R.layout.item_frame_horizontal else R.layout.item_frame_horizontal,
                parent
            ), listener, longListener)
    }



    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(getItem(position), position)
    }

    override fun getItemCount() = currentList.size

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
            bitmap: Bitmap,
            position: Int
        ) =
            with(itemView) {

                itemView.tag = bitmap
                tv_id.text = "$position"
                iv_Image.load(bitmap) {
                    crossfade(true)
                    placeholder(R.mipmap.ic_launcher)
                }

            }
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<Bitmap>() {
        override fun areItemsTheSame(oldItem: Bitmap, newItem: Bitmap): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: Bitmap, newItem: Bitmap): Boolean {
            return oldItem.sameAs(newItem) // == newItem
        }
    }
}

