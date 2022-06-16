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

package com.example.listfolder.annotatescreen

import android.view.LayoutInflater
import android.view.View
import android.view.View.inflate
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.api.load
import com.example.listfolder.R
import com.example.listfolder.repository.NewsItem
import kotlinx.android.extensions.LayoutContainer
import kotlinx.android.synthetic.main.item_view.view.*

class ListFolderRvAdapter(private val listener: (View) -> Unit) :
    ListAdapter<NewsItem, ListFolderRvAdapter.MyViewHolder>(NewsItemItemCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        return MyViewHolder(LayoutInflater.from(parent.context).inflate(R.layout.item_view, parent, false), listener)
//        inflate()
    }

    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    override fun getItemCount() = currentList.size

    inner class MyViewHolder(override val containerView: View, listener: (View) -> Unit) :
        RecyclerView.ViewHolder(containerView),
        LayoutContainer {

        init {
            itemView.setOnClickListener(listener)
        }

        fun bind(newsItem: NewsItem) =
            with(itemView) {
                itemView.tag = newsItem
                tvPath.text = newsItem.path
                tvTime.text = newsItem.modifiedTime
                ivFolderBitmap.load(drawableRes = R.mipmap.ic_launcher)
                ibWorkedOn.load(drawableRes = R.drawable.annotated_64)
//                tvTitle.text = newsItem.title
//                tvDescription.text = newsItem.description
//                ivThumbnail.load(newsItem.imageUrl) {
//                    crossfade(true)
//                    placeholder(R.mipmap.ic_launcher)
//                }
            }
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<NewsItem>() {
        override fun areItemsTheSame(oldItem: NewsItem, newItem: NewsItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: NewsItem, newItem: NewsItem): Boolean {
            return oldItem.path == newItem.path
        }
    }
}

