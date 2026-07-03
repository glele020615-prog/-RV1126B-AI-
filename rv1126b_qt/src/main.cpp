#include "mainwindow.h"

#include <QApplication>
#include <QMetaType>
#include <QImage>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    qRegisterMetaType<QImage>("QImage");

    MainWindow w;
    w.show();

    return app.exec();
}
