package com.example.listfolder.repository

data class NewsItem(

//    val description: String,
    val path: String, // path to folder/ file
    val modifiedTime: String,
    val image: String,
    val isWorkedOn: Boolean
//    val imageUrl: String,
//    val publishedAt: Long
)
