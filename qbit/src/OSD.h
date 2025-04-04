#ifndef OSD_H
#define OSD_H

#include <QDialog>
#include <QLabel>
#include <QHBoxLayout>
#include <QTimer>

class OSD : public QDialog {
    Q_OBJECT

private:
    QTimer *hideOSDTimer;
    QHBoxLayout *hboxLayout;
    QLabel *icon;
    QLabel *msg;

public:
    OSD(QWidget *parent=0);
    ~OSD();

public slots:
    void display(const QString& message);
};

#endif