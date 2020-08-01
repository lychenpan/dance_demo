package org.wedance.mobile.database

import androidx.room.ColumnInfo
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity
data class User(
    @PrimaryKey val uid: Int,
    val videoid: Int
)

data class Keypoints(
    val x: String,
    val y: String,
    val score: String
)
//  x: "x1,x2,....x17"

//a frame's estimation results
data class vedio(
    @Embedded val keypoints1: Keypoints,
    @Embedded val keypoints2: Keypoints
//    @Embedded val keypoints3: Keypoints
//    @Embedded val keypoints4: Keypoints,
//    @Embedded val keypoints5: Keypoints,
)