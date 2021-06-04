name := "mbmc-projekt-si"

version := "0.1"

scalaVersion := "2.12.12"

idePackagePrefix := Some("si")

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.2"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.1.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.1.2"